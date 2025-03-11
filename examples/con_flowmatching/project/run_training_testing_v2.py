import os
import torch
import logging
import numpy as np
from datetime import datetime
from data_processing.cifar10 import get_data_loaders
from model.velocity_network import VelocityNetwork
from model.resnet_velocity import ResNetVelocity, ImprovedCNF
from model.network import CNF
from training.trainer import CNFTrainer
from testing.evaluator import FIDScore, evaluate_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/logs/training_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_and_evaluate(model_name, model, train_loader, test_loader, device, config):
    """Train and evaluate a model with given configuration"""
    logger.info(f"Starting training for {model_name}")
    
    # Initialize trainer
    trainer = CNFTrainer(
        model,
        train_loader,
        test_loader,
        device,
        lr=config['learning_rate'],
        alpha=config.get('alpha', 0.1)
    )
    
    # Initialize metrics
    fid_calculator = FIDScore(device)
    best_fid = float('inf')
    
    metrics_history = {
        'train_loss': [],
        'eval_loss': [],
        'fid_scores': []
    }
    
    # Training loop
    for epoch in range(config['epochs']):
        # Train epoch
        train_loss = trainer.train_epoch(epoch)
        eval_loss = trainer.evaluate()
        
        metrics_history['train_loss'].append(train_loss)
        metrics_history['eval_loss'].append(eval_loss)
        
        logger.info(f"Epoch {epoch+1}/{config['epochs']}")
        logger.info(f"Training Loss: {train_loss:.6f}")
        logger.info(f"Evaluation Loss: {eval_loss:.6f}")
        
        # Compute FID score every few epochs
        if (epoch + 1) % config['eval_interval'] == 0:
            metrics = evaluate_model(trainer.ema_model, test_loader, fid_calculator, device)
            fid_score = metrics['fid']
            metrics_history['fid_scores'].append(fid_score)
            
            logger.info(f"FID Score: {fid_score:.2f}")
            
            if fid_score < best_fid:
                best_fid = fid_score
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.ema_model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'fid_score': fid_score,
                }, f'experiments/checkpoints/{model_name}_best.pt')
    
    return metrics_history

def main():
    # Configuration
    data_dir = os.path.join('data', 'cifar-10-batches-py')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create necessary directories
    os.makedirs('experiments/logs', exist_ok=True)
    os.makedirs('experiments/checkpoints', exist_ok=True)
    os.makedirs('experiments/results', exist_ok=True)
    
    # Training configurations
    configs = {
        'baseline': {
            'model_type': 'simple',
            'hidden_dims': [512, 512, 512],
            'activation': 'relu',
            'learning_rate': 2e-4,
            'alpha': 0.1,
            'epochs': 100,
            'eval_interval': 5,
            'batch_size': 512
        },
        'improved': {
            'model_type': 'resnet',
            'hidden_dims': [128, 256, 512, 256, 128],
            'activation': 'relu',
            'learning_rate': 2e-4,
            'alpha': 0.1,
            'epochs': 100,
            'eval_interval': 5,
            'batch_size': 512
        }
    }
    
    # Set up data loaders
    logger.info("Setting up data loaders...")
    for config_name, config in configs.items():
        train_loader, test_loader = get_data_loaders(
            data_dir,
            batch_size=config['batch_size'],
            num_workers=4
        )
        
        logger.info(f"\nStarting experiments for {config_name} configuration")
        
        try:
            # Initialize model
            if config['model_type'] == 'simple':
                velocity_net = VelocityNetwork(
                    hidden_dims=config['hidden_dims'],
                    activation=config['activation']
                )
                model = CNF(velocity_net)
            else:
                velocity_net = ResNetVelocity(
                    hidden_dims=config['hidden_dims'],
                    activation=config['activation']
                )
                model = ImprovedCNF(velocity_net)
            
            model = model.to(device)
            
            # Train and evaluate
            metrics_history = train_and_evaluate(
                config_name,
                model,
                train_loader,
                test_loader,
                device,
                config
            )
            
            # Save results
            results = {
                'config': config,
                'metrics': metrics_history
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'experiments/results/{config_name}_{timestamp}.npz'
            np.savez(save_path, **results)
            
            logger.info(f"Completed experiments for {config_name}")
            logger.info(f"Results saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error in {config_name} experiments: {str(e)}")
            continue

if __name__ == "__main__":
    main()