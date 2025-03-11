import torch
import json
import os
from model.vqvae import VectorQuantizedVAE
from data_processing.cifar10 import get_data_loaders
from training.trainer import VQVAETrainer
from testing.evaluator import VQVAEEvaluator

def main():
    # Configuration
    config = {
        'data_dir': 'data',
        'batch_size': 128,
        'num_workers': 4,
        'k_dim': 1024,  # Codebook size
        'z_dim': 256,   # Latent dimension
        'beta': 0.25,   # Commitment loss coefficient
        'learning_rate': 2e-4,
        'num_epochs': 50,  # Train for 50 epochs to get meaningful statistics
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    print(f"Using device: {config['device']}")
    print("Loading data...")

    # Data loading
    train_loader, test_loader = get_data_loaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    print("Creating model...")

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
    print("Starting training...")
    training_results = []

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        # Train epoch
        train_metrics = trainer.train_epoch(epoch)
        print(f"Training metrics: {train_metrics}")

        # Test epoch
        test_metrics = trainer.test_epoch()
        print(f"Testing metrics: {test_metrics}")

        epoch_results = {
            'epoch': epoch + 1,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        training_results.append(epoch_results)

    print("\nTraining completed. Running final evaluation...")

    # Save results
    results = {
        'config': config,
        'training_results': training_results,
    }

    os.makedirs('results', exist_ok=True)
    with open('results/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Save model
    torch.save(model.state_dict(), 'results/model_final.pth')

    print("\nExperiment completed. Results saved in 'results' directory.")

if __name__ == '__main__':
    main()