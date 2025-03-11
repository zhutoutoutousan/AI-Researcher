import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = None

    def forward(self, x):
        identity = self.proj(x) if self.proj else x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + identity)


class VectorQuantizedVAE(nn.Module):
    def __init__(self, k_dim=1024, z_dim=256, beta=0.25):
        super().__init__()
        self.k_dim = k_dim  # Codebook size
        self.z_dim = z_dim  # Latent dimension
        self.beta = beta    # Commitment loss coefficient

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(3, z_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(z_dim // 2, z_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(z_dim, z_dim),
            ResidualBlock(z_dim, z_dim)
        )

        # Codebook (Embedding Layer)
        self.codebook = nn.Embedding(k_dim, z_dim)
        
        # Initialize codebook with uniform distribution
        self.codebook.weight.data.uniform_(-1/k_dim, 1/k_dim)
        
        # Decoder network
        self.decoder = nn.Sequential(
            ResidualBlock(z_dim, z_dim),
            ResidualBlock(z_dim, z_dim),
            nn.ConvTranspose2d(z_dim, z_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(z_dim // 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # EMA tracking variables
        register_buffer = lambda name, tensor: self.register_buffer(name, tensor)
        register_buffer('ema_cluster_size', torch.zeros(k_dim))
        register_buffer('ema_dw', torch.zeros(k_dim, z_dim))
        register_buffer('ema_w', self.codebook.weight.data.clone())
        self.ema_decay = 0.99

    def compute_rotation_matrix(self, encoder_output, codebook_vector):
        """
        Compute rotation matrix using Householder transformation
        that aligns encoder output with codebook vector.
        """
        # Reshape encoder output to 2D tensor (batch*h*w, z_dim)
        b, c, h, w = encoder_output.shape
        encoder_output = encoder_output.permute(0, 2, 3, 1).reshape(-1, c)
        codebook_vector = codebook_vector.permute(0, 2, 3, 1).reshape(-1, c)
        
        # Normalize vectors
        e_norm = F.normalize(encoder_output, dim=-1)
        c_norm = F.normalize(codebook_vector, dim=-1)
        
        # Compute reflection vector
        v = e_norm - c_norm
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        mask = (v_norm > 1e-5).float()
        v = mask * (v / (v_norm + 1e-5)) + (1 - mask) * v
        
        # Compute Householder matrix: I - 2vv^T
        I = torch.eye(v.size(-1), device=v.device)
        vvT = torch.matmul(v.unsqueeze(-1), v.unsqueeze(-2))
        H = I - 2 * vvT
        
        # Reshape back to 4D tensor
        H = H.view(b, h, w, c, c)
        
        return H

    def find_nearest_codebook(self, encoder_output):
        """
        Find nearest codebook vector using rotated encoder output
        to improve gradient propagation.
        """
        b, c, h, w = encoder_output.shape
        flat_encoder_output = encoder_output.permute(0, 2, 3, 1).reshape(-1, c)
        
        # Compute distances
        distances = torch.cdist(flat_encoder_output, self.codebook.weight)
        indices = torch.argmin(distances, dim=1)
        
        # Get codebook vectors
        codebook_vectors = self.codebook(indices)
        
        # Reshape back to match encoder output
        codebook_vectors = codebook_vectors.view(b, h, w, c).permute(0, 3, 1, 2)
        indices = indices.view(b, h, w)
        
        return indices, codebook_vectors

    def update_codebook_ema(self, encoder_outputs, indices):
        """
        Update codebook using Exponential Moving Average (EMA)
        """
        with torch.no_grad():
            b, c, h, w = encoder_outputs.shape
            flat_encoder_outputs = encoder_outputs.permute(0, 2, 3, 1).reshape(-1, c)
            flat_indices = indices.reshape(-1)
            
            # Initialize tracking variables
            onehot = F.one_hot(flat_indices, self.k_dim).float()
            
            # Update cluster size
            cluster_size = onehot.sum(0)
            self.ema_cluster_size.data.mul_(self.ema_decay).add_(
                cluster_size * (1 - self.ema_decay)
            )
            
            # Compute sum of encoded vectors per cluster
            dw = torch.matmul(onehot.t(), flat_encoder_outputs)
            self.ema_dw.data.mul_(self.ema_decay).add_(dw * (1 - self.ema_decay))
            
            # Update codebook vectors
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.k_dim * 1e-5) * n
            new_codebook = self.ema_dw / cluster_size.unsqueeze(-1)
            self.codebook.weight.data.copy_(new_codebook)

    def forward(self, x):
        # Encoder
        z_e = self.encoder(x)
        
        # Quantization
        quantization_indices, z_q = self.find_nearest_codebook(z_e)
        
        # Compute rotation matrix for gradient propagation
        rotation_matrices = self.compute_rotation_matrix(z_e, z_q)
        
        # Apply rotation and rescaling transformation (batch-wise)
        b, c, h, w = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(b*h*w, c)
        z_q_flat = z_q.permute(0, 2, 3, 1).reshape(b*h*w, c)
        rotation_matrices_flat = rotation_matrices.reshape(b*h*w, c, c)
        
        z_q_transformed = torch.bmm(rotation_matrices_flat, z_e_flat.unsqueeze(-1)).squeeze(-1)
        z_q_transformed = z_q_transformed.view(b, h, w, c).permute(0, 3, 1, 2)
        
        # Straight-through estimator with transformed gradient
        z_q_straight_through = z_e + (z_q_transformed - z_e).detach()
        
        # Decoder
        x_recon = self.decoder(z_q_straight_through)
        
        # Losses
        reconstruction_loss = F.mse_loss(x_recon, x)
        codebook_loss = F.mse_loss(z_e.detach(), z_q)
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        
        total_loss = reconstruction_loss + codebook_loss + self.beta * commitment_loss
        
        # Update codebook using EMA
        self.update_codebook_ema(z_e.detach(), quantization_indices)
        
        return {
            'reconstruction': x_recon,
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss,
            'perplexity': self.compute_perplexity(quantization_indices),
            'codebook_indices': quantization_indices
        }
    
    def compute_perplexity(self, indices):
        """
        Compute perplexity to monitor codebook usage
        """
        onehot = F.one_hot(indices.flatten(), self.k_dim).float()
        avg_probs = onehot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity