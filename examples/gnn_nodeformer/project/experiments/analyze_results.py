"""Analyze and visualize experimental results."""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def load_experiment_results():
    """Load all experimental results."""
    results = {}
    results_dir = Path(".")
    
    # Load ablation results
    if (results_dir / "ablation_results.json").exists():
        with open(results_dir / "ablation_results.json", "r") as f:
            results["ablation"] = json.load(f)
    
    # Load comparative results
    if (results_dir / "comparative_results.json").exists():
        with open(results_dir / "comparative_results.json", "r") as f:
            results["comparative"] = json.load(f)
    
    # Load temperature results
    if (results_dir / "temperature_results.json").exists():
        with open(results_dir / "temperature_results.json", "r") as f:
            results["temperature"] = json.load(f)
    
    return results

def plot_ablation_results(results):
    """Plot ablation study results."""
    configs = list(results.keys())
    test_accs = [results[c]["best_test_acc"] for c in configs]
    val_accs = [results[c]["best_val_acc"] for c in configs]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(configs))
    width = 0.35
    
    plt.bar(x - width/2, test_accs, width, label='Test Accuracy')
    plt.bar(x + width/2, val_accs, width, label='Validation Accuracy')
    
    plt.xlabel('Model Configuration')
    plt.ylabel('Accuracy')
    plt.title('Ablation Study Results')
    plt.xticks(x, configs, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ablation_results.png')
    plt.close()

def plot_comparative_results(results):
    """Plot comparative study results across datasets."""
    datasets = list(results.keys())
    mean_test_accs = [results[d]["mean_test_acc"] for d in datasets]
    std_test_accs = [results[d]["std_test_acc"] for d in datasets]
    
    plt.figure(figsize=(10, 6))
    plt.bar(datasets, mean_test_accs, yerr=std_test_accs, capsize=5)
    plt.xlabel('Dataset')
    plt.ylabel('Test Accuracy')
    plt.title('Performance Across Datasets')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('comparative_results.png')
    plt.close()

def plot_temperature_study(results):
    """Plot temperature scheduling results."""
    for strategy, data in results.items():
        if "temperature_history" in data:
            plt.figure(figsize=(10, 6))
            plt.plot(data["temperature_history"], label=f'{strategy} schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Temperature')
            plt.title(f'Temperature Schedule - {strategy}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'temperature_schedule_{strategy}.png')
            plt.close()
    
    # Plot performance comparison
    strategies = list(results.keys())
    test_accs = [results[s]["best_test_acc"] for s in strategies]
    
    plt.figure(figsize=(10, 6))
    plt.bar(strategies, test_accs)
    plt.xlabel('Temperature Strategy')
    plt.ylabel('Test Accuracy')
    plt.title('Performance Across Temperature Strategies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('temperature_comparison.png')
    plt.close()

def analyze_results():
    """Analyze and visualize all experimental results."""
    results = load_experiment_results()
    
    # Generate visualizations
    if "ablation" in results:
        plot_ablation_results(results["ablation"])
    
    if "comparative" in results:
        plot_comparative_results(results["comparative"])
    
    if "temperature" in results:
        plot_temperature_study(results["temperature"])
    
    # Generate summary report
    with open('experiment_summary.txt', 'w') as f:
        f.write("Experimental Results Summary\n")
        f.write("===========================\n\n")
        
        if "ablation" in results:
            f.write("Ablation Study Results:\n")
            for config, data in results["ablation"].items():
                f.write(f"{config}:\n")
                f.write(f"  Test Accuracy: {data['best_test_acc']:.4f}\n")
            f.write("\n")
        
        if "comparative" in results:
            f.write("Dataset Comparison Results:\n")
            for dataset, data in results["comparative"].items():
                f.write(f"{dataset}:\n")
                f.write(f"  Mean Test Accuracy: {data['mean_test_acc']:.4f} ± {data['std_test_acc']:.4f}\n")
            f.write("\n")
        
        if "temperature" in results:
            f.write("Temperature Study Results:\n")
            for strategy, data in results["temperature"].items():
                f.write(f"{strategy}:\n")
                f.write(f"  Test Accuracy: {data['best_test_acc']:.4f}\n")

if __name__ == "__main__":
    analyze_results()