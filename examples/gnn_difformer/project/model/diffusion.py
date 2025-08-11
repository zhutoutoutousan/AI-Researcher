import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusivityTransform(nn.Module):
    """Learnable transformation for computing adaptive diffusivity"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, Z):
        """
        Compute diffusivity matrix from node representations
        Args:
            Z (torch.Tensor): Node representations [N x D]
        Returns:
            torch.Tensor: Diffusivity matrix [N x N] 
        """
        transform_Z = self.transform(Z)
        # Compute pairwise similarity
        sim = torch.matmul(transform_Z, transform_Z.t())
        return F.softmax(sim, dim=1)

class EnergyFunction(nn.Module):
    """Energy function regularizing diffusion process"""
    def __init__(self, lambda_reg=1.0):
        super().__init__()
        self.lambda_reg = lambda_reg
    
    def forward(self, Z, Z_prev):
        """
        Compute energy using reconstruction loss and pairwise regularization
        Args:
            Z (torch.Tensor): Current representations
            Z_prev (torch.Tensor): Previous representations
        Returns:
            torch.Tensor: Computed energy value
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(Z, Z_prev)
        
        # Pairwise distance regularization with concave function
        pdist = torch.cdist(Z, Z)
        reg_term = torch.mean(torch.log(1 + pdist))
        
        return recon_loss + self.lambda_reg * reg_term

class DiffusionLayer(nn.Module):
    """Single layer of diffusion propagation"""
    def __init__(self, hidden_dim, tau=0.1):
        super().__init__()
        self.tau = tau
        self.diffusivity = DiffusivityTransform(hidden_dim)
        
    def forward(self, Z):
        """
        Perform one step of diffusion propagation
        Args:
            Z (torch.Tensor): Current node representations
        Returns:
            torch.Tensor: Updated node representations
        """
        # Compute diffusivity matrix
        S = self.diffusivity(Z)
        
        # Compute Laplacian-like propagation
        Z_prop = torch.matmul(S, Z)
        
        # Update using explicit Euler scheme
        Z_next = Z + self.tau * (Z_prop - Z)
        
        return Z_next

class DiffusionModel(nn.Module):
    """Complete diffusion-based representation learning model"""
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, tau=0.1, lambda_reg=1.0):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.num_layers = num_layers
        
        # Multiple diffusion layers
        self.diffusion_layers = nn.ModuleList([
            DiffusionLayer(hidden_dim, tau) for _ in range(num_layers)
        ])
        
        # Energy function for regularization
        self.energy_fn = EnergyFunction(lambda_reg)
        
        # Final prediction layer
        self.output_proj = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, X, training=True):
        """
        Forward pass through the complete model
        Args:
            X (torch.Tensor): Input features
            training (bool): Whether in training mode
        Returns:
            torch.Tensor: Class logits
            torch.Tensor: Energy value (if training)
        """
        # Initial projection
        Z = self.input_proj(X)
        Z_initial = Z.clone()
        
        # Diffusion propagation
        energy = torch.tensor(0.0, device=Z.device)
        for layer in self.diffusion_layers:
            Z_prev = Z.clone()
            Z = layer(Z)
            
            # Compute energy if training
            if training:
                energy = energy + self.energy_fn(Z, Z_initial)
        
        # Final predictions
        logits = self.output_proj(Z)
        
        if training:
            return logits, energy
        return logits