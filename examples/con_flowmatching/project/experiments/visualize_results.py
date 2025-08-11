import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_experiment_results(results_dir):
    """Load all experiment results from the results directory"""
    results = {}
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                results[filename] = json.load(f)
    return results

def plot_training_curves(metrics_history, save_dir):
    """Plot training curves for loss, FID, and other metrics"""
    metrics = ['train_loss', 'eval_loss', 'fid', 'inception_score', 'sample_entropy']
    epochs = [m['epoch'] for m in metrics_history]
    
    plt.figure(figsize=(15, 10))
    for idx, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, idx)
        values = [m[metric] for m in metrics_history]
        plt.plot(epochs, values, marker='o')
        plt.title(f'{metric.replace("_", " ").title()} vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric.replace("_", " ").title())
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def plot_sensitivity_heatmap(sensitivity_results, save_dir):
    """Plot heatmap of parameter sensitivity"""
    # Extract final FID scores for each configuration
    configs = []
    fid_scores = []
    
    for exp_name, result in sensitivity_results.items():
        config = result['params']
        metrics = result['metrics_history'][-1]  # Get last epoch metrics
        
        configs.append(f"lr={config['lr']}\nwd={config['weight_decay']}\nema={config['ema_decay']}\nalpha={config['alpha']}")
        fid_scores.append(metrics['fid'])
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        np.array(fid_scores).reshape(-1, 1),
        annot=True,
        fmt='.2f',
        yticklabels=configs,
        xticklabels=['FID Score'],
        cmap='viridis'
    )
    plt.title('Parameter Sensitivity Analysis (FID Scores)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sensitivity_heatmap.png'))
    plt.close()

def plot_ablation_comparison(ablation_results, save_dir):
    """Plot comparison of different model variants"""
    variants = list(ablation_results.keys())
    metrics = ['fid', 'inception_score', 'sample_entropy']
    
    plt.figure(figsize=(15, 5))
    for idx, metric in enumerate(metrics):
        plt.subplot(1, 3, idx + 1)
        
        values = []
        for variant in variants:
            final_metrics = ablation_results[variant]['metrics_history'][-1]
            values.append(final_metrics[metric])
        
        plt.bar(variants, values)
        plt.title(f'{metric.replace("_", " ").title()} Comparison')
        plt.xticks(rotation=45)
        plt.ylabel(metric.replace("_", " ").title())
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_comparison.png'))
    plt.close()

def main():
    # Create visualization directory
    vis_dir = 'experiments/visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load results
    results_dir = 'experiments/results'
    results = load_experiment_results(results_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(vis_dir, f'analysis_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Process and visualize each type of result
    for filename, result in results.items():
        if 'ablation' in filename:
            plot_ablation_comparison(result, save_dir)
        elif 'sensitivity' in filename:
            plot_sensitivity_heatmap(result, save_dir)
        
        # Plot training curves for all experiments
        if 'metrics_history' in result:
            plot_training_curves(
                result['metrics_history'],
                os.path.join(save_dir, filename.replace('.json', ''))
            )

if __name__ == "__main__":
    main()