import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from model.improved_diffusion import ImprovedDiffusionModel
from data_processing.dataset import load_dataset, get_train_val_test_split
from training.train import train_epoch, evaluate
import json
import os
import torch.nn.functional as F

def setup_visualization(log_dir='./logs/visualization'):
    """Setup logging directory for visualization results"""
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def extract_representations(model, data, device):
    """Extract node representations at different layers"""
    model.eval()
    with torch.no_grad():
        x = data.x.to(device)
        representations = {}
        
        # Initial representation
        Z = model.input_proj(x)
        representations['initial'] = Z.cpu().numpy()
        
        # Intermediate layer representations
        Z_current = Z
        for i, layer in enumerate(model.diffusion_layers):
            Z_current = layer(Z_current)
            representations[f'layer_{i+1}'] = Z_current.cpu().numpy()
        
        # Final representation
        representations['final'] = Z_current.cpu().numpy()
        
    return representations

def visualize_representations(repr_dict, labels, save_dir, title):
    """Visualize representations using t-SNE and PCA"""
    # Create visualization subfolder
    vis_dir = os.path.join(save_dir, 'representation_plots')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Setup visualization methods
    vis_methods = {
        'tsne': TSNE(n_components=2, random_state=42),
        'pca': PCA(n_components=2)
    }
    
    metrics = {}
    for layer_name, repr_data in repr_dict.items():
        metrics[layer_name] = {}
        
        # Create figure for this layer
        plt.figure(figsize=(15, 6))
        
        for idx, (method_name, reducer) in enumerate(vis_methods.items(), 1):
            plt.subplot(1, 2, idx)
            
            # Reduce dimensionality
            reduced_data = reducer.fit_transform(repr_data)
            
            # Plot
            scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                                c=labels, cmap='viridis')
            plt.colorbar(scatter)
            plt.title(f'{method_name.upper()} - {layer_name}')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
        
        plt.suptitle(f'{title} - {layer_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'{title}_{layer_name}.png'))
        plt.close()
        
        # Compute clustering metrics
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        try:
            metrics[layer_name]['silhouette'] = silhouette_score(repr_data, labels)
            metrics[layer_name]['calinski'] = calinski_harabasz_score(repr_data, labels)
        except Exception as e:
            print(f"Could not compute metrics for {layer_name}: {e}")
    
    return metrics

def analyze_representation_evolution(model, data, device):
    """Analyze how representations evolve during diffusion process"""
    evol_metrics = {
        'norm': [],
        'cosine_sim': [],
        'entropy': []
    }
    
    model.eval()
    with torch.no_grad():
        x = data.x.to(device)
        Z = model.input_proj(x)
        Z_initial = Z.clone()
        
        # Track initial metrics
        evol_metrics['norm'].append(torch.norm(Z, dim=1).mean().item())
        evol_metrics['cosine_sim'].append(F.cosine_similarity(Z, Z_initial).mean().item())
        
        # Track evolution through layers
        for layer in model.diffusion_layers:
            Z = layer(Z)
            
            # Compute metrics
            evol_metrics['norm'].append(torch.norm(Z, dim=1).mean().item())
            evol_metrics['cosine_sim'].append(F.cosine_similarity(Z, Z_initial).mean().item())
            
            # Compute entropy of pairwise similarities
            sim_matrix = torch.matmul(F.normalize(Z, dim=1), F.normalize(Z, dim=1).t())
            entropy = -(sim_matrix * torch.log(sim_matrix + 1e-10)).mean().item()
            evol_metrics['entropy'].append(entropy)
    
    return evol_metrics

def plot_evolution_metrics(evol_metrics, save_dir, title):
    """Plot representation evolution metrics"""
    plt.figure(figsize=(15, 5))
    
    # Plot evolution metrics
    for idx, (metric_name, values) in enumerate(evol_metrics.items(), 1):
        plt.subplot(1, 3, idx)
        plt.plot(values, marker='o')
        plt.title(f'{metric_name.replace("_", " ").title()} Evolution')
        plt.xlabel('Layer')
        plt.ylabel(metric_name)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{title}_evolution_metrics.png'))
    plt.close()

def main():
    # Configuration
    device = torch.device('cuda:2')  # Use GPU 2
    log_dir = setup_visualization()
    
    # Model configuration
    config = {
        'hidden_dim': 64,
        'num_layers': 2,
        'tau': 0.1,
        'lambda_reg': 2.0,  # Increased regularization
        'dropout': 0.3      # Increased dropout
    }
    
    visualization_results = {}
    
    for dataset_name in ['Cora', 'CiteSeer']:
        try:
            # Load and preprocess data
            data, num_features, num_classes = load_dataset(dataset_name)
            
            # Initialize model
            model = ImprovedDiffusionModel(
                input_dim=num_features,
                hidden_dim=config['hidden_dim'],
                num_classes=num_classes,
                num_layers=config['num_layers'],
                tau=config['tau'],
                lambda_reg=config['lambda_reg'],
                dropout=config['dropout']
            ).to(device)
            
            # Train model
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            best_val_acc = 0
            for epoch in range(50):
                train_loss, train_acc = train_epoch(model, data, optimizer, device)
            
            # Extract and visualize representations
            representations = extract_representations(model, data, device)
            metrics = visualize_representations(
                representations, 
                data.y.cpu().numpy(),
                log_dir,
                dataset_name
            )
            
            # Analyze representation evolution
            evol_metrics = analyze_representation_evolution(model, data, device)
            plot_evolution_metrics(evol_metrics, log_dir, dataset_name)
            
            # Store results
            visualization_results[dataset_name] = {
                'layer_metrics': metrics,
                'evolution_metrics': evol_metrics
            }
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
    
    # Save comprehensive results
    with open(os.path.join(log_dir, 'visualization_results.json'), 'w') as f:
        json.dump(visualization_results, f, indent=2)

if __name__ == "__main__":
    main()