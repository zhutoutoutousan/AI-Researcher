import torch
import torch.nn as nn
import torch.nn.functional as F

class VelocityNetwork(nn.Module):
    def __init__(self, input_dim=3072, hidden_dims=None, activation='relu'):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]
        self.input_dim = input_dim
        
        # Get activation function
        self.activation = getattr(nn.functional, activation.lower())
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dims[0]),
            getattr(nn, 'ReLU' if activation == 'relu' else 'Tanh')(),
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        
        # Build main network layers
        layers = []
        input_size = input_dim + hidden_dims[0]  # Concatenated with time embedding
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                getattr(nn, 'ReLU' if activation == 'relu' else 'Tanh')(),
            ])
            input_size = hidden_dim
            
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x, t):
        x = x.view(x.shape[0], -1)  # Flatten the image
        t_embedding = self.time_embed(t.view(-1, 1))
        hidden = torch.cat([x, t_embedding], dim=1)
        hidden = self.network(hidden)
        v = self.output_layer(hidden)
        return v.view(x.shape[0], 3, 32, 32)  # Reshape back to image