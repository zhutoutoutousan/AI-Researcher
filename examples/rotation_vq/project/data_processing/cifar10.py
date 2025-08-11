import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.transform = transform
        self.train = train
        
        if self.train:
            self.data = []
            self.targets = []
            for i in range(1, 6):
                data_batch = unpickle(f"{data_dir}/cifar-10-batches-py/data_batch_{i}")
                self.data.append(data_batch[b'data'])
                self.targets.extend(data_batch[b'labels'])
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        else:
            test_batch = unpickle(f"{data_dir}/cifar-10-batches-py/test_batch")
            self.data = test_batch[b'data'].reshape(-1, 3, 32, 32)
            self.targets = test_batch[b'labels']
            
        self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC format
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, target

def get_data_loaders(data_dir, batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = CIFAR10Dataset(data_dir, train=True, transform=transform)
    test_dataset = CIFAR10Dataset(data_dir, train=False, transform=transform)
    
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