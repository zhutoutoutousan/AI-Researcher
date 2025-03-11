import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torchvision.utils as vutils

class VQVAETrainer:
    def __init__(self, model, train_loader, test_loader, device, lr=2e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_codebook_loss = 0
        total_commit_loss = 0
        total_perplexity = 0
        
        pbar = tqdm(self.train_loader)
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            results = self.model(data)
            loss = results['loss']
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += results['reconstruction_loss'].item()
            total_codebook_loss += results['codebook_loss'].item()
            total_commit_loss += results['commitment_loss'].item()
            total_perplexity += results['perplexity'].item()
            
            if batch_idx % 100 == 0:
                # Save sample reconstructions
                with torch.no_grad():
                    sample_size = min(8, data.size(0))
                    comparison = torch.cat([
                        data[:sample_size],
                        results['reconstruction'][:sample_size]
                    ])
                    vutils.save_image(
                        comparison.cpu(),
                        f'reconstruct_epoch{epoch}_batch{batch_idx}.png',
                        normalize=True,
                        nrow=sample_size
                    )
            
            pbar.set_description(
                f'Epoch {epoch}, Loss: {loss.item():.4f}, '
                f'Recon: {results["reconstruction_loss"].item():.4f}, '
                f'Codebook: {results["codebook_loss"].item():.4f}, '
                f'Commit: {results["commitment_loss"].item():.4f}, '
                f'Perplexity: {results["perplexity"].item():.1f}'
            )
        
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches,
            'codebook_loss': total_codebook_loss / n_batches,
            'commitment_loss': total_commit_loss / n_batches,
            'perplexity': total_perplexity / n_batches
        }
    
    def test_epoch(self):
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_codebook_loss = 0
        total_commit_loss = 0
        total_perplexity = 0
        
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                results = self.model(data)
                
                total_loss += results['loss'].item()
                total_recon_loss += results['reconstruction_loss'].item()
                total_codebook_loss += results['codebook_loss'].item()
                total_commit_loss += results['commitment_loss'].item()
                total_perplexity += results['perplexity'].item()
        
        n_batches = len(self.test_loader)
        return {
            'test_loss': total_loss / n_batches,
            'test_recon_loss': total_recon_loss / n_batches,
            'test_codebook_loss': total_codebook_loss / n_batches,
            'test_commitment_loss': total_commit_loss / n_batches,
            'test_perplexity': total_perplexity / n_batches
        }