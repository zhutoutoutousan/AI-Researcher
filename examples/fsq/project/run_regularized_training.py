import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from model.vae import FiniteScalarQuantizedVAE
from data_processing.dataset import get_data_loaders
from training.trainer import Trainer
import json
import os
import logging

# Setup logging
logging.basicConfig(filename='regularized_training.log',
                   level=logging.INFO,
                   format='%(asctime)s - %(message)s')

class RegularizedQuantizationLoss(nn.Module):
    def __init__(self, beta=0.25, entropy_weight=0.1):
        super().__init__()
        self.beta = beta
        self.entropy_weight = entropy_weight
    
    def entropy_regularization(self, quantized_z):
        # Compute distribution of quantized values
        hist = torch.histc(quantized_z, bins=100)
        probs = hist / hist.sum()
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        return entropy
    
    def forward(self, z, quantized_z, codebook):
        # Standard VQ losses
        commitment_loss = F.mse_loss(z, quantized_z.detach())
        codebook_loss = F.mse_loss(quantized_z, z.detach())
        
        # Entropy regularization
        entropy_reg = self.entropy_regularization(quantized_z)
        
        # L2 regularization on codebook
        l2_reg = torch.norm(codebook, p=2)
        
        total_loss = commitment_loss + self.beta * codebook_loss - self.entropy_weight * entropy_reg + 0.01 * l2_reg
        
        return total_loss, {
            'commitment': commitment_loss.item(),
            'codebook': codebook_loss.item(),
            'entropy': entropy_reg.item(),
            'l2': l2_reg.item()
        }

def main():
    # Set device
    device = torch.cuda.current_device()
    logging.info(f"Using device: {device}")

    # Configuration
    config = {
        'batch_size': 32,
        'num_workers': 2,
        'latent_dim': 32,
        'hidden_dim': 32,
        'num_levels': 8,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'beta': 0.25,
        'entropy_weight': 0.1,
        'gradient_clip_val': 1.0
    }

    # Data loaders
    train_loader, test_loader = get_data_loaders(
        data_dir='data',
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Initialize model
    model = FiniteScalarQuantizedVAE(
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        num_levels=config['num_levels']
    ).to(device)

    # Initialize optimizer and trainer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    trainer = Trainer(model, train_loader, test_loader, optimizer, device)

    # Custom loss function
    criterion = RegularizedQuantizationLoss(
        beta=config['beta'],
        entropy_weight=config['entropy_weight']
    )

    history = {
        'train_loss': [], 'test_loss': [],
        'commitment_loss': [], 'codebook_loss': [],
        'entropy_reg': [], 'l2_reg': [],
        'fid': []
    }

    for epoch in range(config['num_epochs']):
        logging.info(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        model.train()

        # Training loop
        train_metrics = {
            'loss': 0.0,
            'commitment': 0.0,
            'codebook': 0.0,
            'entropy': 0.0,
            'l2': 0.0
        }

        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()

            # Forward pass
            encoded = model.encoder(images)
            quantized, quant_info = model.quantizer(encoded)
            reconstructed = model.decoder(quantized)

            # Compute losses
            loss, loss_components = criterion(encoded, quantized, model.quantizer.codebook)

            # Update metrics
            train_metrics['loss'] += loss.item()
            for k, v in loss_components.items():
                train_metrics[k] += v

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_val'])
            optimizer.step()

            if batch_idx % 100 == 0:
                logging.info(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

        # Average metrics
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)

        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Evaluation
        test_metrics = trainer.evaluator.evaluate()

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['test_loss'].append(test_metrics['loss'])
        history['commitment_loss'].append(train_metrics['commitment'])
        history['codebook_loss'].append(train_metrics['codebook'])
        history['entropy_reg'].append(train_metrics['entropy'])
        history['l2_reg'].append(train_metrics['l2'])
        history['fid'].append(test_metrics['fid'])

        # Log metrics
        logging.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logging.info(f"Test Loss: {test_metrics['loss']:.4f}")
        logging.info(f"FID Score: {test_metrics['fid']:.4f}")
        logging.info(f"Commitment Loss: {train_metrics['commitment']:.4f}")
        logging.info(f"Codebook Loss: {train_metrics['codebook']:.4f}")
        logging.info(f"Entropy Reg: {train_metrics['entropy']:.4f}")
        logging.info(f"L2 Reg: {train_metrics['l2']:.4f}")

        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint_path = f'checkpoints/regularized_epoch_{epoch}.pt'
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, checkpoint_path)
            logging.info(f"Saved checkpoint at epoch {epoch}")

        # Clear GPU cache
        torch.cuda.empty_cache()

    # Save final history
    with open('regularized_training_history.json', 'w') as f:
        json.dump(history, f)
    logging.info("Training history saved to regularized_training_history.json")

    # Final evaluation
    final_metrics = trainer.evaluator.evaluate()
    logging.info("\nFinal Evaluation:")
    logging.info(f"Test Loss: {final_metrics['loss']:.4f}")
    logging.info(f"FID Score: {final_metrics['fid']:.4f}")

if __name__ == '__main__':
    main()