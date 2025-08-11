import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, remove_self_loops

def load_dataset(name="Cora", normalize_features=True):
    """
    Load and preprocess graph dataset
    
    Args:
        name (str): Dataset name (Cora, CiteSeer, or PubMed)
        normalize_features (bool): Whether to normalize node features
        
    Returns:
        data: Processed PyG data object
        num_features: Number of input features
        num_classes: Number of output classes
    """
    # Load dataset
    transform = T.NormalizeFeatures() if normalize_features else None
    dataset = Planetoid("./data", name, transform=transform)
    data = dataset[0]
    
    # Process edge indices
    data.edge_index = remove_self_loops(data.edge_index)[0]
    data.edge_index = add_self_loops(data.edge_index)[0]
    
    return data, dataset.num_features, dataset.num_classes

def get_train_val_test_split(data):
    """
    Get training, validation and test masks
    
    Args:
        data: PyG data object
        
    Returns:
        train_mask: Training mask
        val_mask: Validation mask  
        test_mask: Test mask
    """
    return data.train_mask, data.val_mask, data.test_mask