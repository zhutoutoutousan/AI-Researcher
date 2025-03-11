import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """Residual block for continuous function approximation"""
    def __init__(self, dim, activation='relu'):
        super().__init__()
        self.activation = getattr(F, activation)
        
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        h = self.activation(self.layers(x))
        return x + h

class ResNetVelocity(nn.Module):
    """ResNet-based velocity field parameterization"""
    def __init__(self, hidden_dims=[128, 256, 256, 128], activation='relu'):
        super().__init__()
        self.input_dim = 3072  # 3x32x32 flattened
        self.activation = activation
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dims[0]),
            nn.ReLU() if activation == 'relu' else nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        
        # Input projection
        self.input_proj = nn.Linear(self.input_dim + hidden_dims[0], hidden_dims[0])
        
        # ResNet blocks
        self.res_blocks = nn.ModuleList([
            nn.ModuleList([
                ResBlock(dim, activation) for _ in range(2)
            ]) for dim in hidden_dims
        ])
        
        # Dimension changing layers
        self.dim_layers = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[i+1])
            for i in range(len(hidden_dims)-1)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dims[-1], self.input_dim)
        
    def forward(self, x, t):
        # Flatten image and compute time embedding
        x = x.view(x.shape[0], -1)
        t_emb = self.time_embed(t.view(-1, 1))
        
        # Concatenate and project input
        h = torch.cat([x, t_emb], dim=1)
        h = getattr(F, self.activation)(self.input_proj(h))
        
        # Process through ResNet blocks
        for i, (res_block_pair, dim_layer) in enumerate(zip(self.res_blocks[:-1], self.dim_layers)):
            # Apply residual blocks
            for res_block in res_block_pair:
                h = res_block(h)
            # Change dimension
            h = getattr(F, self.activation)(dim_layer(h))
        
        # Final residual blocks
        for res_block in self.res_blocks[-1]:
            h = res_block(h)
        
        # Project to output dimension and reshape
        v = self.output_proj(h)
        return v.view(x.shape[0], 3, 32, 32)

class ImprovedCNF(nn.Module):
    """Improved Continuous Normalizing Flow with better velocity field parameterization"""
    def __init__(self, velocity_net, sigma_min=0.1, n_steps=100):
        super().__init__()
        self.velocity_net = velocity_net
        self.sigma_min = sigma_min
        self.n_steps = n_steps
        
    def euler_integrate(self, x, t, dt, reverse=False):
        """Euler integration step"""
        sign = -1 if reverse else 1
        v = self.velocity_net(x, t)
        return x + sign * dt * v
    
    def rk4_integrate(self, x, t, dt, reverse=False):
        """RK4 integration step"""
        sign = -1 if reverse else 1
        
        # RK4 stages
        k1 = self.velocity_net(x, t)
        k2 = self.velocity_net(x + dt*k1/2, t + dt/2)
        k3 = self.velocity_net(x + dt*k2/2, t + dt/2)
        k4 = self.velocity_net(x + dt*k3, t + dt)
        
        # Combine stages
        return x + sign * dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def sample(self, n_samples, device, method='rk4'):
        """Generate samples using specified integration method"""
        # Sample from base distribution
        x = torch.randn(n_samples, 3, 32, 32).to(device)
        t = torch.zeros(n_samples, device=device)
        
        dt = 1.0 / self.n_steps
        integrate_fn = self.rk4_integrate if method == 'rk4' else self.euler_integrate
        
        for i in range(self.n_steps):
            t = t + dt
            x = integrate_fn(x, t, dt)
        
        return x
    
    def f_theta(self, t, x_t):
        """Compute f_θ(t, x_t) = x_t + (1-t)v_θ(t, x_t)"""
        v = self.velocity_net(x_t, t)
        t = t.view(-1, 1, 1, 1).expand(-1, 3, 32, 32)
        return x_t + (1-t)*v
    
    def velocity_field(self, t, x_t):
        """Compute velocity field v_θ(t, x_t)"""
        return self.velocity_net(x_t, t)
    
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