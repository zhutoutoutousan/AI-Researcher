import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_path, batch_size=512, num_workers=4):
    """Create data loaders for CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Training data
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Test data
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Convert data loader output to tensor output
    class TensorLoader:
        def __init__(self, dataloader):
            self.dataloader = dataloader
            self.length = len(dataloader)
            
        def __iter__(self):
            for batch in self.dataloader:
                yield batch[0]  # Only return images, not labels
                
        def __len__(self):
            return self.length

    train_tensor_loader = TensorLoader(train_loader)
    test_tensor_loader = TensorLoader(test_loader)

    return train_tensor_loader, test_tensor_loader