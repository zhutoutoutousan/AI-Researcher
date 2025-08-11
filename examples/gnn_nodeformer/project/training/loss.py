"""Implementation of loss functions for graph learning."""

import torch
import torch.nn.functional as F


def supervised_loss(pred, labels, mask=None):
    """Compute supervised classification loss.
    
    Args:
        pred (torch.Tensor): Model predictions
        labels (torch.Tensor): Ground truth labels
        mask (torch.Tensor, optional): Mask for selecting valid nodes
        
    Returns:
        torch.Tensor: Classification loss
    """
    if mask is not None:
        pred = pred[mask]
        labels = labels[mask]
    return F.cross_entropy(pred, labels)


def edge_regularization_loss(graph_structure, adj_matrix, temperature=1.0):
    """Compute edge-level regularization loss.
    
    Args:
        graph_structure (torch.Tensor): Learned graph structure
        adj_matrix (torch.Tensor): Original adjacency matrix
        temperature (float): Temperature parameter
        
    Returns:
        torch.Tensor: Edge regularization loss
    """
    # KL divergence between learned structure and original structure
    adj_matrix = adj_matrix.float()
    log_graph_structure = torch.log(graph_structure + 1e-8)
    kl_div = F.kl_div(
        F.log_softmax(log_graph_structure / temperature, dim=-1),
        F.softmax(adj_matrix / temperature, dim=-1),
        reduction='batchmean'
    )
    return kl_div


def combined_loss(pred, labels, graph_structure, adj_matrix, mask=None, 
                 edge_weight=0.1, temperature=1.0):
    """Compute combined loss with classification and edge regularization.
    
    Args:
        pred (torch.Tensor): Model predictions
        labels (torch.Tensor): Ground truth labels
        graph_structure (torch.Tensor): Learned graph structure
        adj_matrix (torch.Tensor): Original adjacency matrix
        mask (torch.Tensor, optional): Mask for selecting valid nodes
        edge_weight (float): Weight for edge regularization loss
        temperature (float): Temperature for edge regularization
        
    Returns:
        torch.Tensor: Combined loss
        dict: Dictionary containing individual loss components
    """
    # Compute classification loss
    cls_loss = supervised_loss(pred, labels, mask)
    
    # Compute edge regularization loss
    edge_loss = edge_regularization_loss(
        graph_structure, adj_matrix, temperature
    )
    
    # Combine losses
    total_loss = cls_loss + edge_weight * edge_loss
    
    # Return total loss and individual components
    loss_components = {
        'classification': cls_loss.item(),
        'edge_regularization': edge_loss.item(),
        'total': total_loss.item()
    }
    
    return total_loss, loss_components