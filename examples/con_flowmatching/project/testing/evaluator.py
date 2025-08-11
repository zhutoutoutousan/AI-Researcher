import torch
import numpy as np
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import inception_v3
import torch.nn.functional as F

class FIDScore:
    def __init__(self, device):
        self.device = device
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
        
    def calculate_activation_statistics(self, images):
        """Calculate activation statistics (mean & covariance) for given images"""
        batch_size = 32  # Reduced batch size to avoid memory issues
        n_samples = len(images)
        n_batches = n_samples // batch_size + (1 if n_samples % batch_size != 0 else 0)
        
        pred_arr = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch = images[start_idx:end_idx]
            
            with torch.no_grad():
                pred = self.inception(batch)
                if isinstance(pred, tuple):  # Get just the logits if model returns tuple
                    pred = pred[0]
                if pred.ndim > 2:  # If predictions have spatial dimensions
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                pred = pred.squeeze()
                pred_arr.append(pred.cpu().numpy())
                
        pred_arr = np.vstack(pred_arr)
            
        mu = np.mean(pred_arr, axis=0)
        sigma = np.cov(pred_arr, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, real_images, generated_images):
        """Calculate FID score between real and generated images"""
        real_mu, real_sigma = self.calculate_activation_statistics(real_images)
        gen_mu, gen_sigma = self.calculate_activation_statistics(generated_images)
        
        mu1, mu2 = real_mu, gen_mu
        sigma1, sigma2 = real_sigma, gen_sigma
        
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        tr_covmean = np.trace(covmean)
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return float(fid)
    
def evaluate_model(model, test_loader, fid_calculator, device, n_samples=1000):  # Reduced n_samples
    """Evaluate model by generating samples and calculating FID score"""
    model.eval()
    
    # Collect real images
    real_images = []
    with torch.no_grad():
        for batch in test_loader:
            real_images.append(batch)
            if len(torch.cat(real_images)) >= n_samples:
                break
    real_images = torch.cat(real_images)[:n_samples].to(device)
    
    # Generate samples
    with torch.no_grad():
        generated_images = []
        batch_size = 32  # Smaller batch size for generation
        remaining = n_samples
        while remaining > 0:
            curr_batch = min(batch_size, remaining)
            generated_images.append(model.sample(curr_batch, device))
            remaining -= curr_batch
        generated_images = torch.cat(generated_images, dim=0)
    
    # Preprocess images to be in [-1, 1] range if they aren't already
    real_images = (real_images - 0.5) * 2
    generated_images = (generated_images - 0.5) * 2
    
    # Resize images to inception input size (299x299)
    real_images = F.interpolate(real_images, size=(299, 299), mode='bilinear', align_corners=False)
    generated_images = F.interpolate(generated_images, size=(299, 299), mode='bilinear', align_corners=False)
    
    # Calculate FID score
    fid = fid_calculator.calculate_fid(real_images, generated_images)
    
    return {'fid': fid}