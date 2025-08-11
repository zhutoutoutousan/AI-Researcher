import torch
import torch.optim as optim
from model.vae import FiniteScalarQuantizedVAE
from data_processing.dataset import get_data_loaders
from training.trainer import Trainer
import logging
import json
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from time import time

# Setup logging
logging.basicConfig(filename='benchmark.log',
                   level=logging.INFO,
                   format='%(asctime)s - %(message)s')

def run_benchmark_experiment(config_variant):
    device = torch.cuda.current_device()
    logging.info(f"Running benchmark with config variant: {config_variant}")

    # Base configuration
    config = {
        'batch_size': 64,
        'num_workers': 4,
        'latent_dim': 64,
        'hidden_dim': 64,
        'learning_rate': 1e-3,
        'num_epochs': 20  # Reduced epochs for benchmarking
    }
    
    # Update config with variant-specific settings
    config.update(config_variant)

    # Record metrics
    metrics = {
        'train_time': [],
        'memory_usage': [],
        'train_loss': [],
        'test_loss': [],
        'fid_score': []
    }

    # Data loading
    train_loader, test_loader = get_data_loaders(
        data_dir='data',
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Model initialization
    model = FiniteScalarQuantizedVAE(
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        num_levels=config['num_levels']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    trainer = Trainer(model, train_loader, test_loader, optimizer, device)

    # Training with profiling
    for epoch in range(config['num_epochs']):
        start_time = time()
        
        with profile(activities=[ProfilerActivity.CUDA], profile_memory=True) as prof:
            epoch_metrics = trainer.train_epoch(epoch)
        
        # Record metrics
        epoch_time = time() - start_time
        memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        metrics['train_time'].append(epoch_time)
        metrics['memory_usage'].append(memory_usage)
        metrics['train_loss'].append(epoch_metrics['train_loss'])
        metrics['test_loss'].append(epoch_metrics['test_loss'])
        metrics['fid_score'].append(epoch_metrics['fid'])

        # Log progress
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        logging.info(f"Time: {epoch_time:.2f}s")
        logging.info(f"Memory: {memory_usage:.2f}MB")
        logging.info(f"Train Loss: {epoch_metrics['train_loss']:.4f}")
        logging.info(f"Test Loss: {epoch_metrics['test_loss']:.4f}")
        logging.info(f"FID Score: {epoch_metrics['fid']:.4f}")

        torch.cuda.empty_cache()

    return metrics

def main():
    # Define benchmark variants
    benchmark_configs = [
        {'name': 'baseline', 'num_levels': 10},
        {'name': 'high_quantization', 'num_levels': 20},
        {'name': 'low_latent', 'num_levels': 10, 'latent_dim': 32},
        {'name': 'high_latent', 'num_levels': 10, 'latent_dim': 128}
    ]

    benchmark_results = {}
    
    for config in benchmark_configs:
        logging.info(f"\nStarting benchmark for {config['name']}")
        try:
            metrics = run_benchmark_experiment(config)
            
            # Compute summary statistics
            summary = {
                'avg_train_time': np.mean(metrics['train_time']),
                'std_train_time': np.std(metrics['train_time']),
                'avg_memory': np.mean(metrics['memory_usage']),
                'final_train_loss': metrics['train_loss'][-1],
                'final_test_loss': metrics['test_loss'][-1],
                'final_fid': metrics['fid_score'][-1],
                'config': config
            }
            
            benchmark_results[config['name']] = summary
            
            logging.info(f"Completed benchmark for {config['name']}")
            logging.info(f"Average training time: {summary['avg_train_time']:.2f}s")
            logging.info(f"Average memory usage: {summary['avg_memory']:.2f}MB")
            logging.info(f"Final FID score: {summary['final_fid']:.4f}")
            
        except Exception as e:
            logging.error(f"Error in benchmark {config['name']}: {str(e)}")
            continue

    # Save benchmark results
    with open('benchmark_results.json', 'w') as f:
        json.dump(benchmark_results, f, indent=4)
    
    logging.info("Benchmark results saved to benchmark_results.json")

    # Print comparative analysis
    logging.info("\nComparative Analysis:")
    for name, results in benchmark_results.items():
        logging.info(f"\n{name}:")
        logging.info(f"Training Time: {results['avg_train_time']:.2f}s ± {results['std_train_time']:.2f}s")
        logging.info(f"Memory Usage: {results['avg_memory']:.2f}MB")
        logging.info(f"Final Train Loss: {results['final_train_loss']:.4f}")
        logging.info(f"Final Test Loss: {results['final_test_loss']:.4f}")
        logging.info(f"Final FID Score: {results['final_fid']:.4f}")

if __name__ == '__main__':
    main()