import os
import torch
import logging
import json
from itertools import product
from datetime import datetime
from data_processing.cifar10 import get_data_loaders
from model.network import VelocityNetwork, CNF
from training.trainer import CNFTrainer
from testing.evaluator import FIDScore
from testing.scores import InceptionScore, SampleEntropyScore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/logs/sensitivity_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_sensitivity_experiment(params, device, train_loader, test_loader):
    """Run sensitivity analysis experiment with given parameters"""
    logger.info(f"Starting sensitivity experiment with params: {params}")
    
    # Model configuration
    model_config = {
        'hidden_dims': [128, 256, 256, 128],
        'activation': 'relu'
    }
    
    # Initialize model and trainer
    velocity_net = VelocityNetwork(**model_config)
    model = CNF(velocity_net)
    
    trainer = CNFTrainer(
        model,
        train_loader,
        test_loader,
        device,
        lr=params['lr'],
        weight_decay=params['weight_decay'],
        ema_decay=params['ema_decay'],
        alpha=params['alpha']
    )
    
    # Initialize metrics
    fid_calculator = FIDScore(device)
    inception_calculator = InceptionScore(device)
    entropy_calculator = SampleEntropyScore()
    
    # Training loop
    metrics_history = []
    for epoch in range(params['n_epochs']):
        train_loss = trainer.train_epoch(epoch)
        eval_loss = trainer.evaluate()
        
        if (epoch + 1) % params['eval_interval'] == 0:
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'fid': fid_calculator.calculate_fid(trainer.ema_model, test_loader),
                'inception_score': inception_calculator.calculate(trainer.ema_model),
                'sample_entropy': entropy_calculator.calculate(trainer.ema_model)
            }
            metrics_history.append(metrics)
            
            logger.info(f"Epoch {epoch+1}/{params['n_epochs']}")
            logger.info(f"Metrics: {metrics}")
    
    return metrics_history

def main():
    # Configuration
    data_dir = os.path.join('data', 'cifar-10-batches-py')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 512
    
    # Create directories
    os.makedirs('experiments/logs', exist_ok=True)
    os.makedirs('experiments/results', exist_ok=True)
    
    # Set up data loaders
    logger.info("Setting up data loaders...")
    train_loader, test_loader = get_data_loaders(data_dir, batch_size=batch_size, num_workers=4)
    
    # Parameter combinations for sensitivity analysis
    param_grid = {
        'lr': [1e-4, 2e-4, 5e-4],
        'weight_decay': [1e-4, 1e-5],
        'ema_decay': [0.999, 0.9999],
        'alpha': [0.5, 1.0, 2.0],
        'n_epochs': [50],  # Reduced for time efficiency
        'eval_interval': [5]
    }
    
    # Generate all combinations
    param_keys = param_grid.keys()
    param_values = param_grid.values()
    experiments = [dict(zip(param_keys, v)) for v in product(*param_values)]
    
    # Run sensitivity analysis
    results = {}
    for params in experiments:
        experiment_name = f"lr{params['lr']}_wd{params['weight_decay']}_ema{params['ema_decay']}_alpha{params['alpha']}"
        try:
            metrics_history = run_sensitivity_experiment(
                params, device, train_loader, test_loader
            )
            results[experiment_name] = {
                'params': params,
                'metrics_history': metrics_history
            }
        except Exception as e:
            logger.error(f"Experiment {experiment_name} failed: {str(e)}")
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'experiments/results/sensitivity_analysis_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()