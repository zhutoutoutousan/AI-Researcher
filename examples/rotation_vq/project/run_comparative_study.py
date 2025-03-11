import torch
import json
import os
from model.vqvae import VectorQuantizedVAE
from data_processing.cifar10 import get_data_loaders
from training.trainer import VQVAETrainer
from testing.evaluator import VQVAEEvaluator

def run_comparative_study():
    # Base configuration
    base_config = {
        'data_dir': 'data',
        'batch_size': 128,
        'num_workers': 4,
        'z_dim': 256,   # Latent dimension
        'beta': 0.25,   # Fixed commitment loss coefficient
        'num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    # Model variants to compare
    model_configs = {
        'small': {'k_dim': 256, 'learning_rate': 2e-4},
        'medium': {'k_dim': 512, 'learning_rate': 2e-4},
        'large': {'k_dim': 1024, 'learning_rate': 2e-4},
        'xlarge': {'k_dim': 2048, 'learning_rate': 2e-4}
    }

    comparative_results = {}

    for model_size, model_config in model_configs.items():
        print(f"\nRunning experiment with model size = {model_size}")
        
        config = {**base_config, **model_config}
        results_dir = f'results/model_size_{model_size}'
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

        # Save individual results
        results = {
            'config': config,
            'training_results': training_results,
        }
        with open(os.path.join(results_dir, 'experiment_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        # Run comprehensive evaluation
        evaluator = VQVAEEvaluator(
            model, 
            test_loader, 
            reference_stats_path='/workplace/dataset_candidate/cifar10-32x32.npz'
        )
        eval_results = evaluator.run_comprehensive_evaluation()

        comparative_results[model_size] = {
            'final_train_metrics': training_results[-1]['train_metrics'],
            'final_test_metrics': training_results[-1]['test_metrics'],
            'evaluation_metrics': eval_results
        }

    # Save comparative study summary
    with open('results/model_size_comparison_summary.json', 'w') as f:
        json.dump(comparative_results, f, indent=4)

if __name__ == '__main__':
    run_comparative_study()