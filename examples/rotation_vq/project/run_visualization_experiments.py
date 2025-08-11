import torch
import torch.nn as nn
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model.vqvae import VectorQuantizedVAE
from data_processing.cifar10 import get_data_loaders
from torch.utils.tensorboard import SummaryWriter

class GradientFlowTracker(nn.Module):
    def __init__(self, model, log_dir):
        super().__init__()
        self.model = model
        self.writer = SummaryWriter(log_dir)
        self.gradient_history = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(name):
            def fn(grad):
                if grad is not None:
                    self.gradient_history.append({
                        'name': name,
                        'grad_norm': grad.norm().item(),
                        'grad_mean': grad.mean().item(),
                        'grad_std': grad.std().item()
                    })
            return fn

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.hooks.append(param.register_hook(hook_fn(name)))

    def save_gradient_flow(self, epoch):
        if not self.gradient_history:
            return

        for grad_info in self.gradient_history:
            self.writer.add_scalar(
                f'Gradient_Flow/{grad_info["name"]}/norm',
                grad_info['grad_norm'],
                epoch
            )
            self.writer.add_scalar(
                f'Gradient_Flow/{grad_info["name"]}/mean',
                grad_info['grad_mean'],
                epoch
            )
            self.writer.add_scalar(
                f'Gradient_Flow/{grad_info["name"]}/std',
                grad_info['grad_std'],
                epoch
            )

        self.gradient_history = []

def visualize_codebook_evolution(model, epoch, save_dir):
    # Extract codebook vectors
    codebook_vectors = model.codebook.weight.detach().cpu().numpy()
    
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    codebook_2d = tsne.fit_transform(codebook_vectors)
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    plt.scatter(codebook_2d[:, 0], codebook_2d[:, 1], alpha=0.5)
    plt.title(f'Codebook Vector Distribution (Epoch {epoch})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'codebook_distribution_epoch_{epoch}.png'))
    plt.close()

def run_visualization_experiments():
    # Configuration
    config = {
        'data_dir': 'data',
        'batch_size': 128,
        'num_workers': 4,
        'k_dim': 1024,
        'z_dim': 256,
        'beta': 0.25,
        'learning_rate': 2e-4,
        'num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    # Setup directories
    vis_dir = 'results/visualization'
    os.makedirs(vis_dir, exist_ok=True)
    config['log_dir'] = os.path.join(vis_dir, 'logs')
    config['checkpoint_path'] = os.path.join(vis_dir, 'model_final.pth')

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
    ).to(config['device'])

    # Initialize gradient tracker
    gradient_tracker = GradientFlowTracker(model, config['log_dir'])

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training loop with visualization
    training_results = []
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(config['device'])
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = output['loss']

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Save gradient flow information
            gradient_tracker.save_gradient_flow(epoch * len(train_loader) + batch_idx)

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        # Visualize codebook distribution
        visualize_codebook_evolution(model, epoch, os.path.join(vis_dir, 'codebook_evolution'))

        # Save epoch results
        epoch_results = {
            'epoch': epoch + 1,
            'average_loss': epoch_loss / len(train_loader)
        }
        training_results.append(epoch_results)

    # Save visualization experiment results
    results = {
        'config': config,
        'training_results': training_results
    }
    with open(os.path.join(vis_dir, 'visualization_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    run_visualization_experiments()