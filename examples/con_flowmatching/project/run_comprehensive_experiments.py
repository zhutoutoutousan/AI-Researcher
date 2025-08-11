import os
import torch
import logging
from datetime import datetime
from data_processing.cifar10 import get_data_loaders
from model.network import VelocityNetwork, CNF
from model.resnet_velocity import ResNetVelocity, ImprovedCNF
from training.trainer import CNFTrainer
from testing.evaluator import FIDScore, evaluate_model
from experiments.ablation_study import run_experiment as run_ablation
from experiments.sensitivity_analysis import run_sensitivity_experiment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/logs/comprehensive_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Configuration
    data_dir = os.path.join('data', 'cifar-10-batches-py')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 512
    n_epochs = 100  # Extended training for better results
    
    # Set up data loaders
    logger.info("Setting up data loaders...")
    train_loader, test_loader = get_data_loaders(data_dir, batch_size=batch_size, num_workers=4)
    
    # 1. Baseline Model Training
    logger.info("Training baseline model...")
    velocity_net = VelocityNetwork().to(device)
    baseline_model = CNF(velocity_net)
    baseline_trainer = CNFTrainer(baseline_model, train_loader, test_loader, device)
    
    for epoch in range(n_epochs):
        train_loss = baseline_trainer.train_epoch(epoch)
        eval_loss = baseline_trainer.evaluate()
        logger.info(f"Baseline - Epoch {epoch+1}/{n_epochs}")
        logger.info(f"Training Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
    
    # 2. Improved Model Training
    logger.info("Training improved model with ResNet architecture...")
    resnet_velocity = ResNetVelocity().to(device)
    improved_model = ImprovedCNF(resnet_velocity)
    improved_trainer = CNFTrainer(improved_model, train_loader, test_loader, device)
    
    for epoch in range(n_epochs):
        train_loss = improved_trainer.train_epoch(epoch)
        eval_loss = improved_trainer.evaluate()
        logger.info(f"Improved - Epoch {epoch+1}/{n_epochs}")
        logger.info(f"Training Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
    
    # 3. Ablation Studies
    logger.info("Running ablation studies...")
    ablation_configs = {
        'baseline': {
            'model_params': {'hidden_dims': [128, 256, 256, 128], 'activation': 'relu'},
            'training_params': {'lr': 2e-4, 'weight_decay': 1e-4, 'ema_decay': 0.999, 'alpha': 1.0},
            'n_epochs': n_epochs,
            'eval_interval': 5
        },
        'deeper_network': {
            'model_params': {'hidden_dims': [128, 256, 512, 256, 128], 'activation': 'relu'},
            'training_params': {'lr': 2e-4, 'weight_decay': 1e-4, 'ema_decay': 0.999, 'alpha': 1.0},
            'n_epochs': n_epochs,
            'eval_interval': 5
        }
    }
    
    ablation_results = {}
    for name, config in ablation_configs.items():
        logger.info(f"Running ablation study: {name}")
        try:
            result = run_ablation(config, device, train_loader, test_loader, name)
            ablation_results[name] = result
        except Exception as e:
            logger.error(f"Ablation study {name} failed: {str(e)}")
    
    # 4. Sensitivity Analysis
    logger.info("Running sensitivity analysis...")
    sensitivity_params = {
        'lr': 2e-4,
        'weight_decay': 1e-4,
        'ema_decay': 0.999,
        'alpha': 1.0,
        'n_epochs': n_epochs,
        'eval_interval': 5
    }
    
    try:
        sensitivity_results = run_sensitivity_experiment(
            sensitivity_params, device, train_loader, test_loader
        )
    except Exception as e:
        logger.error(f"Sensitivity analysis failed: {str(e)}")
    
    # 5. Final Evaluation
    logger.info("Running final evaluation...")
    fid_calculator = FIDScore(device)
    
    # Evaluate baseline model
    baseline_metrics = evaluate_model(baseline_trainer.ema_model, test_loader, fid_calculator, device)
    logger.info(f"Baseline Model Metrics: {baseline_metrics}")
    
    # Evaluate improved model
    improved_metrics = evaluate_model(improved_trainer.ema_model, test_loader, fid_calculator, device)
    logger.info(f"Improved Model Metrics: {improved_metrics}")
    
    # Save results
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'baseline_metrics': baseline_metrics,
        'improved_metrics': improved_metrics,
        'ablation_results': ablation_results,
        'sensitivity_results': sensitivity_results if 'sensitivity_results' in locals() else None
    }
    
    os.makedirs('experiments/results', exist_ok=True)
    import json
    with open(f'experiments/results/comprehensive_results_{results["timestamp"]}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info("Comprehensive experiments completed. Results saved.")

if __name__ == "__main__":
    main()