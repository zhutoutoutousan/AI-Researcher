import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os

class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        """CIFAR-10 dataset with preprocessing.
        
        Args:
            root_dir (str): Directory containing dataset
            train (bool): If True, use training set
            transform: Optional transform to apply
        """
        self.transform = transform
        self.train = train
        
        if train:
            self.data = []
            for i in range(1, 6):
                with open(os.path.join(root_dir, f'data_batch_{i}'), 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
            self.data = np.vstack(self.data)
        else:
            with open(os.path.join(root_dir, 'test_batch'), 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data = entry['data']
                
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = torch.FloatTensor(self.data) / 255.0
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img

def get_data_loaders(data_dir, batch_size=128, num_workers=4):
    """Create data loaders for training and testing.
    
    Args:
        data_dir (str): Directory containing dataset
        batch_size (int): Batch size
        num_workers (int): Number of worker threads
        
    Returns:
        tuple: Training and test data loaders
    """
    # Create datasets
    train_dataset = CIFAR10Dataset(
        root_dir=os.path.join(data_dir, 'cifar-10-batches-py'),
        train=True
    )
    test_dataset = CIFAR10Dataset(
        root_dir=os.path.join(data_dir, 'cifar-10-batches-py'),
        train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader