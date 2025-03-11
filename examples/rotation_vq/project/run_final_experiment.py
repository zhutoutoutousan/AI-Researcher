import torch
import json
import os
from model.vqvae import VectorQuantizedVAE
from data_processing.cifar10 import get_data_loaders
from training.trainer import VQVAETrainer
from testing.evaluator import VQVAEEvaluator
import logging

# Set up logging
logging.basicConfig(
    filename='results/final_experiment.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def run_final_experiment():
    # Configuration
    config = {
        'data_dir': 'data',
        'batch_size': 128,
        'num_workers': 4,
        'k_dim': 1024,
        'z_dim': 256,
        'beta': 0.25,
        'learning_rate': 2e-4,
        'num_epochs': 100,  # Extended training period
        'device': 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda',  # Try alternate GPU
    }

    logging.info(f"Starting final experiment with config: {config}")
    logging.info(f"Using device: {config['device']}")

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

    # Results directory setup
    os.makedirs('results/final_experiment', exist_ok=True)
    config['log_dir'] = 'results/final_experiment/logs'
    config['checkpoint_path'] = 'results/final_experiment/model_final.pth'

    # Training setup
    trainer = VQVAETrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=config['device'],
        lr=config['learning_rate']
    )

    # Training loop with detailed logging
    epoch_results = []
    best_perplexity = 0
    best_recon_loss = float('inf')

    for epoch in range(config['num_epochs']):
        # Training
        train_metrics = trainer.train_epoch(epoch)
        test_metrics = trainer.test_epoch()

        # Log key metrics
        logging.info(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        logging.info(f"Train Metrics: {train_metrics}")
        logging.info(f"Test Metrics: {test_metrics}")

        # Save best models based on different metrics
        if train_metrics.get('perplexity', 0) > best_perplexity:
            best_perplexity = train_metrics['perplexity']
            torch.save(model.state_dict(), 'results/final_experiment/best_perplexity_model.pth')
            logging.info(f"New best perplexity: {best_perplexity}")

        if train_metrics.get('reconstruction_loss', float('inf')) < best_recon_loss:
            best_recon_loss = train_metrics['reconstruction_loss']
            torch.save(model.state_dict(), 'results/final_experiment/best_recon_model.pth')
            logging.info(f"New best reconstruction loss: {best_recon_loss}")

        # Save epoch results
        epoch_results.append({
            'epoch': epoch + 1,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        })

        # Save results periodically
        if (epoch + 1) % 10 == 0:
            results = {
                'config': config,
                'epoch_results': epoch_results,
                'best_perplexity': best_perplexity,
                'best_recon_loss': best_recon_loss
            }
            with open('results/final_experiment/experiment_results.json', 'w') as f:
                json.dump(results, f, indent=4)

    # Final evaluation
    logging.info("\nRunning final comprehensive evaluation...")
    evaluator = VQVAEEvaluator(model, test_loader)
    final_eval_results = evaluator.run_comprehensive_evaluation()
    logging.info(f"Final Evaluation Results: {final_eval_results}")

    # Save final results
    final_results = {
        'config': config,
        'epoch_results': epoch_results,
        'best_perplexity': best_perplexity,
        'best_recon_loss': best_recon_loss,
        'final_evaluation': final_eval_results
    }
    with open('results/final_experiment/final_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)

    logging.info("Final experiment completed successfully.")
    return final_results

if __name__ == '__main__':
    run_final_experiment()