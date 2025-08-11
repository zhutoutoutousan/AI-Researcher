import torch
import torch.optim as optim
from model.vae import FiniteScalarQuantizedVAE
from data_processing.dataset import get_data_loaders
from training.trainer import Trainer
import json
import os
import logging

# Setup logging
logging.basicConfig(filename='adaptive_quantization.log',
                   level=logging.INFO,
                   format='%(asctime)s - %(message)s')

class AdaptiveQuantization:
    def __init__(self, config):
        self.base_levels = config['base_levels']
        self.max_levels = config['max_levels']
        self.adaptation_threshold = config['adaptation_threshold']
        self.levels = self.base_levels

    def adapt_levels(self, quantization_loss):
        if quantization_loss > self.adaptation_threshold:
            self.levels = min(self.levels + 2, self.max_levels)
            return True
        return False

def main():
    # Set device
    device = torch.cuda.current_device()
    logging.info(f"Using device: {device}")

    # Set hyperparameters with adaptive quantization
    config = {
        'batch_size': 32,  # Reduced batch size
        'num_workers': 2,
        'latent_dim': 32,  # Reduced latent dimension
        'hidden_dim': 32,  # Reduced hidden dimension
        'base_levels': 5,
        'max_levels': 15,
        'learning_rate': 1e-3,
        'num_epochs': 50,
        'adaptation_threshold': 1000.0,
        'grad_clip_value': 1.0
    }

    # Create data loaders
    logging.info("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(
        data_dir='data',
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Initialize adaptive quantization
    adaptive_quant = AdaptiveQuantization(config)

    # Initialize model
    logging.info("Initializing model...")
    model = FiniteScalarQuantizedVAE(
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        num_levels=adaptive_quant.levels
    ).to(device)

    # Initialize optimizer with gradient clipping
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    trainer = Trainer(model, train_loader, test_loader, optimizer, device)

    history = {'train_loss': [], 'test_loss': [], 'quant_levels': [], 'fid': []}

    for epoch in range(config['num_epochs']):
        logging.info(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        logging.info(f"Current quantization levels: {adaptive_quant.levels}")

        # Train and evaluate
        train_metrics = trainer.train_epoch(epoch)

        # Clear GPU cache
        torch.cuda.empty_cache()
        
        test_metrics = trainer.evaluator.evaluate()

        # Log metrics
        logging.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logging.info(f"Test Loss: {test_metrics['loss']:.4f}")
        logging.info(f"FID Score: {test_metrics['fid']:.4f}")
        logging.info(f"Quantization Loss: {train_metrics['quant_loss']:.4f}")

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['test_loss'].append(test_metrics['loss'])
        history['quant_levels'].append(adaptive_quant.levels)
        history['fid'].append(test_metrics['fid'])

        # Adapt quantization levels
        if adaptive_quant.adapt_levels(train_metrics['quant_loss']):
            # Create new model with updated levels
            new_model = FiniteScalarQuantizedVAE(
                latent_dim=config['latent_dim'],
                hidden_dim=config['hidden_dim'],
                num_levels=adaptive_quant.levels
            ).to(device)
            
            # Transfer learnable parameters
            new_model.load_state_dict(model.state_dict(), strict=False)
            model = new_model
            trainer = Trainer(model, train_loader, test_loader, optimizer, device)
            
            # Update optimizer
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            logging.info(f"Adapted quantization levels to: {adaptive_quant.levels}")

        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint_path = f'checkpoints/adaptive_quant_epoch_{epoch}.pt'
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'quant_levels': adaptive_quant.levels,
                'history': history
            }, checkpoint_path)
            logging.info(f"Saved checkpoint at epoch {epoch}")

        # Clear GPU cache
        torch.cuda.empty_cache()

    # Save final training history
    with open('adaptive_quantization_history.json', 'w') as f:
        json.dump(history, f)
    logging.info("Training history saved to adaptive_quantization_history.json")

    # Final evaluation
    final_metrics = trainer.evaluator.evaluate()
    logging.info("\nFinal Evaluation:")
    logging.info(f"Test Loss: {final_metrics['loss']:.4f}")
    logging.info(f"FID Score: {final_metrics['fid']:.4f}")

if __name__ == '__main__':
    main()