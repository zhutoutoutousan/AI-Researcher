import torch
import torch.nn.functional as F
from pathlib import Path

class VQVAEEvaluator:
    def __init__(self, model, test_loader):
        """
        Initialize VQ-VAE model evaluator
        
        Args:
            model (nn.Module): Trained VQ-VAE model
            test_loader (DataLoader): Test dataset loader
        """
        self.model = model
        self.test_loader = test_loader
        self.device = next(model.parameters()).device
    
    def compute_reconstruction_metrics(self):
        """
        Compute reconstruction quality metrics
        """
        self.model.eval()
        reconstruction_losses = []
        
        with torch.no_grad():
            for images in self.test_loader:
                if isinstance(images, (list, tuple)):
                    images = images[0]
                images = images.to(self.device)
                
                # Reconstruction
                model_output = self.model(images)
                reconstructed_images = model_output['reconstruction']
                
                # Reconstruction Loss
                recon_loss = F.mse_loss(reconstructed_images, images, reduction='mean')
                reconstruction_losses.append(recon_loss.item())
        
        return {
            'avg_reconstruction_loss': sum(reconstruction_losses) / len(reconstruction_losses)
        }
    
    def compute_codebook_metrics(self):
        """
        Analyze codebook utilization and diversity
        """
        self.model.eval()
        codebook_usage = torch.zeros(self.model.k_dim).to(self.device)
        total_encodings = 0
        
        with torch.no_grad():
            for images in self.test_loader:
                if isinstance(images, (list, tuple)):
                    images = images[0]
                images = images.to(self.device)
                
                # Get codebook indices
                z_e = self.model.encoder(images)
                indices = self.model.find_nearest_codebook(z_e)[0]
                
                # Count usage
                for idx in range(self.model.k_dim):
                    codebook_usage[idx] += (indices == idx).sum().item()
                total_encodings += indices.numel()
        
        # Compute usage statistics
        usage_rate = codebook_usage / total_encodings
        active_codes = (codebook_usage > 0).sum().item()
        perplexity = torch.exp(-torch.sum(usage_rate * torch.log(usage_rate + 1e-10)))
        
        return {
            'codebook_utilization': active_codes / self.model.k_dim,
            'perplexity': perplexity.item()
        }
    
    def compute_sample_quality_metrics(self, num_samples=1000):
        """
        Compute metrics for generated samples
        """
        self.model.eval()
        reconstruction_quality = []
        
        with torch.no_grad():
            for i in range(0, num_samples, self.test_loader.batch_size):
                # Take a batch of real images
                images = next(iter(self.test_loader))[0][:min(self.test_loader.batch_size, num_samples - i)]
                images = images.to(self.device)
                
                # Reconstruct
                reconstructions = self.model(images)['reconstruction']
                
                # Compute quality
                quality = F.mse_loss(reconstructions, images, reduction='none').mean([1, 2, 3])
                reconstruction_quality.extend(quality.tolist())
        
        return {
            'avg_sample_quality': sum(reconstruction_quality) / len(reconstruction_quality)
        }
    
    def run_comprehensive_evaluation(self):
        """
        Run comprehensive model evaluation
        """
        reconstruction_metrics = self.compute_reconstruction_metrics()
        codebook_metrics = self.compute_codebook_metrics()
        sample_metrics = self.compute_sample_quality_metrics()
        
        return {
            **reconstruction_metrics,
            **codebook_metrics,
            **sample_metrics
        }