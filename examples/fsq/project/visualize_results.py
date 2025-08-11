import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from sklearn.manifold import TSNE
import torch
import os
import logging

# Setup logging
logging.basicConfig(filename='visualization.log',
                   level=logging.INFO,
                   format='%(asctime)s - %(message)s')

def plot_training_curves(history_file, output_dir='visualization'):
    """Plot training and validation curves."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    plt.close()
    
    # Plot FID scores
    plt.figure(figsize=(12, 8))
    plt.plot(history['fid'], label='FID Score')
    plt.xlabel('Epoch')
    plt.ylabel('FID Score')
    plt.title('FID Score Evolution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'fid_scores.png'))
    plt.close()
    
    logging.info("Training curves plotted successfully")

def plot_latent_space(model, test_loader, output_dir='visualization'):
    """Plot t-SNE visualization of latent space."""
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for images, label in test_loader:
            encoded = model.encoder(images.cuda())
            latent_vectors.append(encoded.cpu().numpy())
            labels.append(label.numpy())
    
    latent_vectors = np.concatenate(latent_vectors)
    labels = np.concatenate(labels)
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_tsne = tsne.fit_transform(latent_vectors.reshape(latent_vectors.shape[0], -1))
    
    # Plot t-SNE results
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Latent Space')
    plt.savefig(os.path.join(output_dir, 'latent_tsne.png'))
    plt.close()
    
    logging.info("Latent space visualization completed")

def plot_quantization_analysis(model, test_loader, output_dir='visualization'):
    """Analyze and plot quantization characteristics."""
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    quantized_values = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            encoded = model.encoder(images.cuda())
            quantized = model.quantizer(encoded)[0]
            quantized_values.append(quantized.cpu().numpy())
    
    quantized_values = np.concatenate(quantized_values)
    
    # Plot quantization level distribution
    plt.figure(figsize=(12, 8))
    plt.hist(quantized_values.flatten(), bins=50)
    plt.title('Distribution of Quantized Values')
    plt.xlabel('Quantized Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'quantization_dist.png'))
    plt.close()
    
    # Plot codebook utilization heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(model.quantizer.codebook.cpu().numpy(), cmap='viridis')
    plt.title('Codebook Utilization Heatmap')
    plt.savefig(os.path.join(output_dir, 'codebook_heatmap.png'))
    plt.close()
    
    logging.info("Quantization analysis plots generated")

def plot_hierarchical_analysis(history_file, output_dir='visualization'):
    """Plot analysis of hierarchical quantization results."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    if 'level_utilization' in history:
        # Plot level utilization
        level_util = np.array(history['level_utilization'])
        plt.figure(figsize=(12, 8))
        for i in range(level_util.shape[1]):
            plt.plot(level_util[:, i], label=f'Level {i+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Utilization')
        plt.title('Hierarchical Level Utilization')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'hierarchical_utilization.png'))
        plt.close()
        
        logging.info("Hierarchical analysis plots generated")

def main():
    # Temperature annealing results
    plot_training_curves('temperature_annealing_history.json')
    
    # Regularized training results
    plot_training_curves('regularized_training_history.json')
    
    # Hierarchical quantization results
    plot_training_curves('hierarchical_quantization_history.json')
    plot_hierarchical_analysis('hierarchical_quantization_history.json')
    
    # Load best model and visualize latent space
    if os.path.exists('checkpoints/best_model.pt'):
        model = torch.load('checkpoints/best_model.pt')
        test_loader = get_data_loaders(data_dir='data', batch_size=32)['validation']
        
        plot_latent_space(model, test_loader)
        plot_quantization_analysis(model, test_loader)
        
        logging.info("All visualizations completed successfully")

if __name__ == '__main__':
    main()