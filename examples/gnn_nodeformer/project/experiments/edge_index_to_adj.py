"""Utility functions for converting between edge_index and adjacency matrix."""

import torch

def edge_index_to_adj(edge_index, num_nodes):
    """Convert edge_index to adjacency matrix.
    
    Args:
        edge_index (torch.Tensor): Edge index tensor of shape [2, E]
        num_nodes (int): Number of nodes in the graph
    
    Returns:
        torch.Tensor: Adjacency matrix of shape [N, N]
    """
    device = edge_index.device
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=device)
    adj[edge_index[0], edge_index[1]] = 1
    return adj

def convert_edge_index_batch(edge_index_batch, batch_size, num_nodes_per_batch):
    """Convert batch of edge_indices to batch of adjacency matrices.
    
    Args:
        edge_index_batch (torch.Tensor): Batch of edge indices
        batch_size (int): Batch size
        num_nodes_per_batch (int): Number of nodes per graph in batch
    
    Returns:
        torch.Tensor: Batch of adjacency matrices
    """
    device = edge_index_batch.device
    adj_batch = torch.zeros(batch_size, num_nodes_per_batch, num_nodes_per_batch,
                          dtype=torch.float, device=device)
    
    for i in range(batch_size):
        # Extract edge_index for current graph
        start_idx = i * num_nodes_per_batch
        end_idx = (i + 1) * num_nodes_per_batch
        mask = (edge_index_batch[0] >= start_idx) & (edge_index_batch[0] < end_idx)
        
        # Convert to local indexing
        edge_index = edge_index_batch[:, mask]
        edge_index = edge_index - start_idx
        
        # Convert to adjacency matrix
        adj_batch[i] = edge_index_to_adj(edge_index, num_nodes_per_batch)
    
    return adj_batch