import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3
from scipy.stats import entropy

class InceptionScore:
    def __init__(self, device, splits=10):
        self.device = device
        self.splits = splits
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()

    def calculate(self, images):
        """Calculate inception score"""
        batch_size = 32
        n_samples = len(images)
        n_batches = n_samples // batch_size + (1 if n_samples % batch_size != 0 else 0)
        probs = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch = images[start_idx:end_idx]

            # Resize to inception size
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)

            with torch.no_grad():
                pred = self.inception(batch)
                if isinstance(pred, tuple):
                    pred = pred[0]
                prob = F.softmax(pred, dim=1)
                probs.append(prob.cpu().numpy())

        probs = np.concatenate(probs, axis=0)
        
        # Split predictions into groups
        split_scores = []
        for k in range(self.splits):
            part = probs[k * (len(probs) // self.splits): 
                        (k + 1) * (len(probs) // self.splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
            kl = np.mean(np.sum(kl, axis=1))
            split_scores.append(np.exp(kl))

        return {
            'inception_mean': np.mean(split_scores),
            'inception_std': np.std(split_scores)
        }

class SampleEntropyScore:
    def __init__(self, n_neighbors=5, eps=1e-8):
        self.n_neighbors = n_neighbors
        self.eps = eps

    def calculate(self, images):
        """Calculate sample entropy score"""
        # Flatten images
        flat_images = images.view(images.size(0), -1).cpu().numpy()
        
        # Compute pairwise distances
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(flat_images)
        
        # Sort distances and get n_neighbors nearest neighbors
        sorted_distances = np.sort(distances, axis=1)
        neighbor_distances = sorted_distances[:, 1:self.n_neighbors+1]  # exclude self
        
        # Compute entropy
        mean_distances = np.mean(neighbor_distances, axis=1)
        entropy_score = -np.mean(np.log(mean_distances + self.eps))
        
        return {'sample_entropy': float(entropy_score)}