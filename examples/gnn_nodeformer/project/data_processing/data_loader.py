"""Data loading utilities for graph datasets."""

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, add_self_loops
import os

def load_dataset(dataset_name):
    """Load a dataset from PyTorch Geometric.
    
    Args:
        dataset_name (str): Name of the dataset ('Cora', 'CiteSeer', or 'PubMed')
    
    Returns:
        tuple: (graph, (num_features, num_classes))
    """
    # Set path for dataset
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
        graph = dataset[0]
        
        # Preprocess graph structure
        graph.edge_index = remove_self_loops(graph.edge_index)[0]
        graph.edge_index = add_self_loops(graph.edge_index, num_nodes=graph.x.size(0))[0]
        
        return graph, (dataset.num_features, dataset.num_classes)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")