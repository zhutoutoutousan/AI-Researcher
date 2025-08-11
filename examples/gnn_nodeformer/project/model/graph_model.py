"""Implementation of the graph neural network model with kernelized Gumbel-Softmax."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .kernelized_gumbel import KernelizedGumbelSoftmax


class GraphConvLayer(nn.Module):
    """Graph convolution layer with learned graph structure."""
    
    def __init__(self, in_features, out_features, dropout=0.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, graph_structure):
        """Forward pass through the graph convolution layer.
        
        Args:
            x (torch.Tensor): Node feature matrix
            graph_structure (torch.Tensor): Learned graph structure
            
        Returns:
            torch.Tensor: Updated node embeddings
        """
        # Message passing with learned graph structure
        x = torch.matmul(graph_structure, x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class KernelizedGraphLearner(nn.Module):
    """Graph learning model with kernelized Gumbel-Softmax operator."""
    
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, 
                 temperature=0.4, dropout=0.5):
        """Initialize the graph learning model.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            num_classes (int): Number of output classes
            num_layers (int): Number of graph convolution layers
            temperature (float): Temperature for Gumbel-Softmax
            dropout (float): Dropout probability
        """
        super().__init__()
        self.num_layers = num_layers
        
        # Graph structure learning components
        self.graph_learner = KernelizedGumbelSoftmax(temperature=temperature)
        self.structure_learner = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph convolution layers
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvLayer(input_dim, hidden_dim, dropout))
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvLayer(hidden_dim, hidden_dim, dropout))
        self.layers.append(GraphConvLayer(hidden_dim, num_classes, dropout))
        
        self.dropout = nn.Dropout(dropout)
        
    def compute_graph_structure(self, x, adj_matrix=None):
        """Compute the graph structure using kernelized Gumbel-Softmax.
        
        Args:
            x (torch.Tensor): Node feature matrix
            adj_matrix (torch.Tensor, optional): Original adjacency matrix
            
        Returns:
            torch.Tensor: Learned graph structure
        """
        # Generate logits for graph structure
        h = self.structure_learner(x)
        logits = torch.matmul(h, h.transpose(-2, -1))
        # Apply kernelized Gumbel-Softmax
        graph_structure = self.graph_learner(logits, adj_matrix)
        return graph_structure
        
    def forward(self, x, adj_matrix=None):
        """Forward pass of the graph learning model.
        
        Args:
            x (torch.Tensor): Input node features
            adj_matrix (torch.Tensor, optional): Original adjacency matrix
            
        Returns:
            torch.Tensor: Node classification logits
            torch.Tensor: Learned graph structure
        """
        # Learn graph structure
        graph_structure = self.compute_graph_structure(x, adj_matrix)
        
        # Graph convolution with learned structure
        for i in range(self.num_layers):
            x = self.layers[i](x, graph_structure)
            if i < self.num_layers - 1:
                x = F.relu(x)
                
        return x, graph_structure