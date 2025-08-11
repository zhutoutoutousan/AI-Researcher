import torch
import json
import os
from model.vqvae import VectorQuantizedVAE
from data_processing.cifar10 import get_data_loaders
from training.trainer import VQVAETrainer
from testing.evaluator import VQVAEEvaluator

def run_ablation_studies():
    # Base configurations
    base_config = {
        'data_dir': 'data',
        'batch_size': 128,
        'num_workers': 4,
        'k_dim': 1024,  # Codebook size
        'z_dim': 256,   # Latent dimension
        'num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    # Ablation study 1: Beta coefficient sensitivity analysis
    beta_values = [0.1, 0.25, 0.5, 1.0, 2.0]
    beta_results = {}

    for beta in beta_values:
        print(f"\nRunning experiment with beta = {beta}")
        
        config = base_config.copy()
        config['beta'] = beta
        config['learning_rate'] = 2e-4
        results_dir = f'results/beta_{beta}'
        os.makedirs(results_dir, exist_ok=True)
        config['log_dir'] = os.path.join(results_dir, 'logs')
        config['checkpoint_path'] = os.path.join(results_dir, 'model_final.pth')

        # Data loading
        train_loader, test_loader = get_data_loaders(
            config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )

        # Model initialization
        model = VectorQuantizedVAE(
            k_dim=config['k_dim'],
            z_dim=config['z_dim'],
            beta=config['beta']
        )

        # Training setup
        trainer = VQVAETrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=config['device'],
            lr=config['learning_rate']
        )

        # Training loop
        training_results = []
        for epoch in range(config['num_epochs']):
            train_metrics = trainer.train_epoch(epoch)
            test_metrics = trainer.test_epoch()

            epoch_results = {
                'epoch': epoch + 1,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            }
            training_results.append(epoch_results)

        # Save results
        results = {
            'config': config,
            'training_results': training_results,
        }
        with open(os.path.join(results_dir, 'experiment_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        # Store final metrics
        beta_results[beta] = {
            'final_train_metrics': training_results[-1]['train_metrics'],
            'final_test_metrics': training_results[-1]['test_metrics']
        }

    # Save ablation study summary
    with open('results/beta_ablation_summary.json', 'w') as f:
        json.dump(beta_results, f, indent=4)

if __name__ == '__main__':
    run_ablation_studies()