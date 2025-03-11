"""Enhanced implementation with improved architectural components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .kernelized_gumbel import KernelizedGumbelSoftmax

class MultiHeadAttention(nn.Module):
    """Multi-head attention for graph structure learning."""
    
    def __init__(self, input_dim, num_heads=4):
        super().__init__()
        # Project input to a dimension that's divisible by num_heads
        self.num_heads = num_heads
        self.head_dim = max(1, input_dim // num_heads)
        self.total_head_dim = self.head_dim * num_heads
        self.input_dim = input_dim
        
        # Projection layers
        self.in_proj = nn.Linear(input_dim, self.total_head_dim)
        self.q_linear = nn.Linear(self.total_head_dim, self.total_head_dim)
        self.k_linear = nn.Linear(self.total_head_dim, self.total_head_dim)
        self.v_linear = nn.Linear(self.total_head_dim, self.total_head_dim)
        self.out_proj = nn.Linear(self.total_head_dim, input_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input
        x = self.in_proj(x)
        
        # Linear transformations and reshape
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.total_head_dim)
        
        # Project back to original dimension
        output = self.out_proj(context)
        output = self.norm(output + x)  # Residual connection
        
        return output, attn

class EnhancedGraphConvLayer(nn.Module):
    """Enhanced graph convolution layer with multi-head attention."""
    
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.5):
        super().__init__()
        self.attention = MultiHeadAttention(in_features, num_heads)
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, graph_structure):
        # Apply attention first
        attended_x, _ = self.attention(x)
        
        # Message passing with learned graph structure
        x = torch.matmul(graph_structure, attended_x)
        x = self.linear(x)
        x = self.norm(x)  # Layer normalization
        x = self.dropout(x)
        return x

class EnhancedGraphLearner(nn.Module):
    """Enhanced graph learning model with improved components."""
    
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2,
                 temperature=0.4, dropout=0.5, num_heads=4):
        super().__init__()
        self.num_layers = num_layers
        
        # Graph structure learning components
        self.graph_learner = KernelizedGumbelSoftmax(temperature=temperature)
        self.structure_learner = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph convolution layers with residual connections
        self.layers = nn.ModuleList()
        self.layers.append(EnhancedGraphConvLayer(input_dim, hidden_dim, num_heads, dropout))
        
        for _ in range(num_layers - 2):
            self.layers.append(EnhancedGraphConvLayer(hidden_dim, hidden_dim, num_heads, dropout))
        
        self.layers.append(EnhancedGraphConvLayer(hidden_dim, num_classes, num_heads, dropout))
        
        # Additional components
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) if i == 0 
            else nn.Linear(hidden_dim, num_classes) if i == num_layers - 1
            else nn.Identity()
            for i in range(num_layers)
        ])
        
    def compute_graph_structure(self, x, adj_matrix=None):
        # Generate logits for graph structure
        h = self.structure_learner(x)
        h = self.layer_norm(h)  # Additional normalization
        
        logits = torch.matmul(h, h.transpose(-2, -1))
        # Apply kernelized Gumbel-Softmax
        graph_structure = self.graph_learner(logits, adj_matrix)
        return graph_structure
    
    def forward(self, x, adj_matrix=None):
        # Learn graph structure
        graph_structure = self.compute_graph_structure(x, adj_matrix)
        
        # Graph convolution with skip connections and layer normalization
        for i in range(self.num_layers):
            skip = self.skip_connections[i](x)
            x = self.layers[i](x, graph_structure)
            
            if i < self.num_layers - 1:
                x = F.relu(x)
            
            # Add skip connection if shapes match
            if skip.shape == x.shape:
                x = x + skip
        
        return x, graph_structure