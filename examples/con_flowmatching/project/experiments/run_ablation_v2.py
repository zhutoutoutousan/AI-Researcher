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
        logging.FileHandler('experiments/logs/ablation_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_ablation_experiment(config, train_loader, test_loader, device):
    """Run a single ablation experiment"""
    logger.info(f"Starting ablation experiment with config: {config}")
    
    try:
        # Initialize model based on configuration
        if config['architecture'] == 'simple':
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
        
        # Initialize trainer
        trainer = CNFTrainer(
            model,
            train_loader,
            test_loader,
            device,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            ema_decay=config['ema_decay']
        )
        
        # Initialize metrics
        fid_calculator = FIDScore(device)
        metrics_history = {
            'train_loss': [],
            'eval_loss': [],
            'fid_scores': []
        }
        
        # Training loop
        for epoch in range(config['epochs']):
            train_loss = trainer.train_epoch(epoch)
            eval_loss = trainer.evaluate()
            
            metrics_history['train_loss'].append(train_loss)
            metrics_history['eval_loss'].append(eval_loss)
            
            if (epoch + 1) % config['eval_interval'] == 0:
                metrics = evaluate_model(trainer.ema_model, test_loader, fid_calculator, device)
                metrics_history['fid_scores'].append(metrics['fid'])
                
                logger.info(f"Epoch {epoch+1}/{config['epochs']}")
                logger.info(f"Training Loss: {train_loss:.6f}, Eval Loss: {eval_loss:.6f}")
                logger.info(f"FID Score: {metrics['fid']:.2f}")
        
        return metrics_history
        
    except Exception as e:
        logger.error(f"Error in ablation experiment: {str(e)}")
        return None

def main():
    # Configuration
    data_dir = os.path.join('data', 'cifar-10-batches-py')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create directories
    os.makedirs('experiments/logs', exist_ok=True)
    os.makedirs('experiments/results', exist_ok=True)
    
    # Ablation configurations
    ablation_configs = {
        'baseline': {
            'architecture': 'simple',
            'hidden_dims': [512, 512, 512],
            'activation': 'relu',
            'learning_rate': 2e-4,
            'weight_decay': 1e-4,
            'ema_decay': 0.999,
            'epochs': 50,
            'eval_interval': 5,
            'batch_size': 512,
            'alpha': 0.1
        },
        'deep_network': {
            'architecture': 'simple',
            'hidden_dims': [512, 512, 512, 512, 512],
            'activation': 'relu',
            'learning_rate': 2e-4,
            'weight_decay': 1e-4,
            'ema_decay': 0.999,
            'epochs': 50,
            'eval_interval': 5,
            'batch_size': 512,
            'alpha': 0.1
        },
        'tanh_activation': {
            'architecture': 'simple',
            'hidden_dims': [512, 512, 512],
            'activation': 'tanh',
            'learning_rate': 2e-4,
            'weight_decay': 1e-4,
            'ema_decay': 0.999,
            'epochs': 50,
            'eval_interval': 5,
            'batch_size': 512,
            'alpha': 0.1
        },
        'resnet_architecture': {
            'architecture': 'resnet',
            'hidden_dims': [128, 256, 512, 256, 128],
            'activation': 'relu',
            'learning_rate': 2e-4,
            'weight_decay': 1e-4,
            'ema_decay': 0.999,
            'epochs': 50,
            'eval_interval': 5,
            'batch_size': 512,
            'alpha': 0.1
        }
    }
    
    # Load data
    train_loader, test_loader = get_data_loaders(data_dir, batch_size=512, num_workers=4)
    
    # Run experiments
    results = {}
    for name, config in ablation_configs.items():
        logger.info(f"\nStarting ablation study: {name}")
        metrics = run_ablation_experiment(config, train_loader, test_loader, device)
        
        if metrics is not None:
            results[name] = {
                'config': config,
                'metrics': metrics
            }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'experiments/results/ablation_results_v2_{timestamp}.npz'
    np.savez(save_path, **results)
    logger.info(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()