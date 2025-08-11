import torch
import torch.optim as optim
from model.vae import FiniteScalarQuantizedVAE
from data_processing.dataset import get_data_loaders
from training.trainer import Trainer
import json
import os
import logging
import torch.nn.functional as F
import numpy as np

# Setup logging
logging.basicConfig(filename='hierarchical_quantization.log',
                   level=logging.INFO,
                   format='%(asctime)s - %(message)s')

class HierarchicalQuantization(nn.Module):
    def __init__(self, base_levels=5, level_multiplier=2, num_hierarchies=3, embedding_dim=32):
        super().__init__()
        self.base_levels = base_levels
        self.level_multiplier = level_multiplier
        self.num_hierarchies = num_hierarchies
        self.embedding_dim = embedding_dim
        
        # Create hierarchical quantizers
        self.quantizers = nn.ModuleList([
            FiniteScalarQuantization(
                num_levels=base_levels * (level_multiplier ** i),
                embedding_dim=embedding_dim
            ) for i in range(num_hierarchies)
        ])
        
        # Projection layers between hierarchies
        self.projections = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim)
            for _ in range(num_hierarchies - 1)
        ])
        
        self.combination_weights = nn.Parameter(
            torch.ones(num_hierarchies) / num_hierarchies
        )
        
    def forward(self, z):
        quantized_outputs = []
        current_z = z
        
        # Forward pass through hierarchies
        for i in range(self.num_hierarchies):
            quantized = self.quantizers[i](current_z)
            quantized_outputs.append(quantized)
            
            if i < self.num_hierarchies - 1:
                current_z = self.projections[i](quantized)
        
        # Weighted combination of hierarchies
        weights = F.softmax(self.combination_weights, dim=0)
        combined_output = sum(w * q for w, q in zip(weights, quantized_outputs))
        
        return combined_output, quantized_outputs

def main():
    # Set device and seed
    device = torch.cuda.current_device()
    torch.manual_seed(42)
    logging.info(f"Using device: {device}")

    # Configuration
    config = {
        'batch_size': 32,
        'num_workers': 2,
        'latent_dim': 32,
        'hidden_dim': 32,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'base_levels': 5,
        'level_multiplier': 2,
        'num_hierarchies': 3,
        'gradient_clip_val': 1.0
    }

    # Data loaders
    train_loader, test_loader = get_data_loaders(
        data_dir='data',
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Initialize models
    hierarchical_quantizer = HierarchicalQuantization(
        base_levels=config['base_levels'],
        level_multiplier=config['level_multiplier'],
        num_hierarchies=config['num_hierarchies'],
        embedding_dim=config['latent_dim']
    ).to(device)

    model = FiniteScalarQuantizedVAE(
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        quantizer=hierarchical_quantizer
    ).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(hierarchical_quantizer.parameters()),
        lr=config['learning_rate']
    )

    history = {
        'train_loss': [], 'test_loss': [],
        'train_recon_loss': [], 'test_recon_loss': [],
        'train_quant_loss': [], 'test_quant_loss': [],
        'level_utilization': [], 'fid': []
    }

    for epoch in range(config['num_epochs']):
        logging.info(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        model.train()
        
        train_metrics = trainer.train_epoch(epoch)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        test_metrics = trainer.evaluator.evaluate()

        # Log level utilization
        level_utils = []
        for i, quantizer in enumerate(hierarchical_quantizer.quantizers):
            utilization = quantizer.get_codebook_utilization()
            level_utils.append(utilization)
            logging.info(f"Level {i+1} utilization: {utilization:.4f}")

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['test_loss'].append(test_metrics['loss'])
        history['train_recon_loss'].append(train_metrics['recon_loss'])
        history['test_recon_loss'].append(test_metrics['recon_loss'])
        history['train_quant_loss'].append(train_metrics['quant_loss'])
        history['test_quant_loss'].append(test_metrics['quant_loss'])
        history['level_utilization'].append(level_utils)
        history['fid'].append(test_metrics['fid'])

        # Log metrics
        logging.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logging.info(f"Test Loss: {test_metrics['loss']:.4f}")
        logging.info(f"FID Score: {test_metrics['fid']:.4f}")
        
        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint_path = f'checkpoints/hierarchical_quant_epoch_{epoch}.pt'
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
    with open('hierarchical_quantization_history.json', 'w') as f:
        json.dump(history, f)
    logging.info("Training history saved to hierarchical_quantization_history.json")

    # Final evaluation
    final_metrics = trainer.evaluator.evaluate()
    logging.info("\nFinal Evaluation:")
    logging.info(f"Test Loss: {final_metrics['loss']:.4f}")
    logging.info(f"FID Score: {final_metrics['fid']:.4f}")

if __name__ == '__main__':
    main()