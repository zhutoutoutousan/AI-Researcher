import torch
import torch.nn as nn

class VelocityNetwork(nn.Module):
    """Network that parameterizes the velocity field v_θ(t, x)"""
    def __init__(self, input_dim=3072, hidden_dim=512):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x, t):
        x = x.view(x.shape[0], -1)  # Flatten the image
        t_emb = self.time_embed(t.view(-1, 1))
        h = torch.cat([x, t_emb], dim=1)
        v = self.net(h)
        return v.view(x.shape[0], 3, 32, 32)  # Reshape back to image

class CNF(nn.Module):
    """Continuous Normalizing Flow model"""
    def __init__(self, velocity_net, sigma_min=0.1, n_steps=100):
        super().__init__()
        self.velocity_net = velocity_net
        self.sigma_min = sigma_min
        self.n_steps = n_steps
        
    def f_theta(self, t, x_t):
        """Compute f_θ(t, x_t) = x_t + (1-t)v_θ(t, x_t)"""
        v = self.velocity_net(x_t, t)
        t = t.view(-1, 1, 1, 1).expand(-1, 3, 32, 32)
        return x_t + (1-t)*v
        
    def velocity_field(self, t, x_t):
        """Compute velocity field v_θ(t, x_t)"""
        return self.velocity_net(x_t, t)
    
    def sample(self, n_samples, device):
        """Generate samples using Euler method"""
        # Sample from base distribution
        x = torch.randn(n_samples, 3, 32, 32).to(device)
        
        dt = 1.0 / self.n_steps
        for i in range(self.n_steps):
            t = torch.ones(n_samples, device=device) * (i * dt)
            v = self.velocity_field(t, x)
            x = x + dt * v
            
        return x

    def forward(self, x_0, x_1, t):
        """Forward pass computing velocity consistency loss"""
        dt = 0.01  # Small time step for consistency loss
        t_next = torch.clamp(t + dt, max=1.0)
        
        # Current points
        t_exp = t.view(-1, 1, 1, 1)
        t_next_exp = t_next.view(-1, 1, 1, 1)
        x_t = t_exp * x_1 + (1 - t_exp) * x_0
        
        # Next points
        x_next = t_next_exp * x_1 + (1 - t_next_exp) * x_0
        
        # Compute f_theta and velocity terms
        f_t = self.f_theta(t, x_t)
        f_next = self.f_theta(t_next, x_next)
        
        v_t = self.velocity_field(t, x_t)
        v_next = self.velocity_field(t_next, x_next)
        
        return f_t, f_next, v_t, v_next