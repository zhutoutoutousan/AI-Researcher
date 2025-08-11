import os
import torch
import logging
import json
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
        logging.FileHandler('experiments/logs/ablation_study.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_experiment(config, device, train_loader, test_loader, experiment_name):
    """Run a single experiment with given configuration"""
    logger.info(f"Starting experiment: {experiment_name}")
    
    # Initialize model with config
    velocity_net = VelocityNetwork(**config['model_params'])
    model = CNF(velocity_net)
    
    # Initialize trainer
    trainer = CNFTrainer(
        model, 
        train_loader, 
        test_loader, 
        device,
        **config['training_params']
    )
    
    # Initialize metrics
    fid_calculator = FIDScore(device)
    inception_calculator = InceptionScore(device)
    entropy_calculator = SampleEntropyScore()
    
    # Training loop
    metrics_history = []
    for epoch in range(config['n_epochs']):
        train_loss = trainer.train_epoch(epoch)
        eval_loss = trainer.evaluate()
        
        if (epoch + 1) % config['eval_interval'] == 0:
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'fid': fid_calculator.calculate_fid(trainer.ema_model, test_loader),
                'inception_score': inception_calculator.calculate(trainer.ema_model),
                'sample_entropy': entropy_calculator.calculate(trainer.ema_model)
            }
            metrics_history.append(metrics)
            
            logger.info(f"Epoch {epoch+1}/{config['n_epochs']}")
            logger.info(f"Metrics: {metrics}")
    
    # Save results
    results = {
        'experiment_name': experiment_name,
        'config': config,
        'metrics_history': metrics_history
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'experiments/results/{experiment_name}_{timestamp}.json'
    os.makedirs('experiments/results', exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

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
    
    # Define experiment configurations
    experiments = {
        'baseline': {
            'model_params': {
                'hidden_dims': [128, 256, 256, 128],
                'activation': 'relu'
            },
            'training_params': {
                'lr': 2e-4,
                'weight_decay': 1e-4,
                'ema_decay': 0.999,
                'alpha': 1.0
            },
            'n_epochs': 100,
            'eval_interval': 5
        },
        'deeper_network': {
            'model_params': {
                'hidden_dims': [128, 256, 512, 256, 128],
                'activation': 'relu'
            },
            'training_params': {
                'lr': 2e-4,
                'weight_decay': 1e-4,
                'ema_decay': 0.999,
                'alpha': 1.0
            },
            'n_epochs': 100,
            'eval_interval': 5
        },
        'tanh_activation': {
            'model_params': {
                'hidden_dims': [128, 256, 256, 128],
                'activation': 'tanh'
            },
            'training_params': {
                'lr': 2e-4,
                'weight_decay': 1e-4,
                'ema_decay': 0.999,
                'alpha': 1.0
            },
            'n_epochs': 100,
            'eval_interval': 5
        }
    }
    
    # Run experiments
    results = {}
    for exp_name, config in experiments.items():
        try:
            results[exp_name] = run_experiment(
                config, device, train_loader, test_loader, exp_name
            )
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {str(e)}")
            continue
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'experiments/results/combined_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()