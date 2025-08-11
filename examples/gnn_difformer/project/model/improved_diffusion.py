import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ImprovedDiffusivityTransform(nn.Module):
    """Enhanced diffusivity transformation with multi-head attention mechanism"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.proj_q = nn.Linear(hidden_dim, hidden_dim)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Z):
        B = Z.size(0)  # Batch size
        
        # Linear transformations for efficiency
        Z_transformed = self.transform(Z)
        
        # Compute attention scores directly with no reshaping
        Q = self.proj_q(Z_transformed)
        K = self.proj_k(Z_transformed)
        
        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        return attn

class ImprovedEnergyFunction(nn.Module):
    """Enhanced energy function with adaptive regularization"""
    def __init__(self, hidden_dim, lambda_reg=1.0):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.adaptive_weight = nn.Parameter(torch.ones(1))
        self.scale_factor = nn.Parameter(torch.ones(1))
        
    def concave_function(self, x):
        """Smooth concave function with controlled scaling"""
        return self.scale_factor * torch.log(1 + x)
        
    def forward(self, Z, Z_prev, epoch=None):
        # Reconstruction loss with normalized features
        Z_norm = F.normalize(Z, p=2, dim=-1)
        Z_prev_norm = F.normalize(Z_prev, p=2, dim=-1)
        recon_loss = F.mse_loss(Z_norm, Z_prev_norm)
        
        # Enhanced pairwise regularization with memory-efficient implementation
        Z_norms = torch.sum(Z * Z, dim=-1, keepdim=True)
        # Compute distances efficiently
        dists = Z_norms + Z_norms.transpose(-2, -1) - 2 * torch.matmul(Z, Z.transpose(-2, -1))
        dists = torch.clamp(dists, min=0.0)  # Ensure non-negative
        reg_term = torch.mean(self.concave_function(dists))
        
        # Adaptive regularization strength
        if epoch is not None:
            lambda_schedule = min(1.0, epoch / 50)  # Gradually increase regularization
            current_lambda = self.lambda_reg * lambda_schedule
        else:
            current_lambda = self.lambda_reg
            
        return recon_loss + current_lambda * self.adaptive_weight * reg_term

class ResidualDiffusionLayer(nn.Module):
    """Diffusion layer with residual connections and layer normalization"""
    def __init__(self, hidden_dim, tau=0.1, dropout=0.1):
        super().__init__()
        self.tau = tau
        self.diffusivity = ImprovedDiffusivityTransform(hidden_dim, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Z):
        # Residual connection and layer normalization
        Z_norm = self.layer_norm1(Z)
        S = self.diffusivity(Z_norm)
        
        # Memory-efficient propagation
        Z_prop = torch.matmul(S, Z_norm)
        Z_prop = self.dropout(Z_prop)
        
        # First residual connection
        Z_res1 = Z + self.tau * (Z_prop - Z)
        
        # Second normalization and residual
        Z_res2 = self.layer_norm2(Z_res1)
        
        return Z_res2

class ImprovedDiffusionModel(nn.Module):
    """Enhanced diffusion model with improved regularization and architecture"""
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, tau=0.1, 
                 lambda_reg=1.0, dropout=0.1):
        super().__init__()
        
        # Input projection with normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.num_layers = num_layers

        # Diffusion layers with residual connections
        self.diffusion_layers = nn.ModuleList([
            ResidualDiffusionLayer(hidden_dim, tau, dropout)
            for _ in range(num_layers)
        ])

        # Improved energy function
        self.energy_fn = ImprovedEnergyFunction(hidden_dim, lambda_reg)

        # Output projection with skip connection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenated with input
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, X, training=True, epoch=None):
        # Initial projection
        Z = self.input_proj(X)
        Z_initial = Z.clone()

        # Diffusion propagation with energy tracking
        energy = torch.tensor(0.0, device=Z.device)
        intermediate_representations = []
        
        for layer in self.diffusion_layers:
            Z_prev = Z.clone()
            Z = layer(Z)
            intermediate_representations.append(Z)

            if training:
                energy = energy + self.energy_fn(Z, Z_initial, epoch)

        # Combine initial and final representations
        Z_combined = torch.cat([Z, Z_initial], dim=-1)
        logits = self.output_proj(Z_combined)

        if training:
            return logits, energy
        return logits