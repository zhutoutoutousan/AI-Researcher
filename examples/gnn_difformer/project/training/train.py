import torch
import torch.nn.functional as F
from torch import optim
import logging

def accuracy(y_pred, y_true):
    """
    Calculate accuracy between predictions and ground truth

    Args:
        y_pred (torch.Tensor): Predicted logits
        y_true (torch.Tensor): Ground truth labels

    Returns:
        float: Accuracy score
    """
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)

def train_epoch(model, data, optimizer, device, epoch=None):
    """
    Train for one epoch

    Args:
        model: Diffusion model
        data: PyG data object
        optimizer: PyTorch optimizer
        device: Computing device
        epoch: Current epoch number (optional)

    Returns:
        float: Training loss
        float: Training accuracy
    """
    model.train()

    # Move data to device
    x = data.x.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)

    # Forward pass
    optimizer.zero_grad()
    logits, energy = model(x, training=True, epoch=epoch)

    # Calculate losses
    cls_loss = F.cross_entropy(logits[train_mask], y[train_mask])
    total_loss = cls_loss + 0.1 * energy  # Weight energy term

    # Backward pass
    total_loss.backward()
    optimizer.step()

    # Calculate metrics
    acc = accuracy(logits[train_mask], y[train_mask])

    return total_loss.item(), acc

@torch.no_grad()
def evaluate(model, data, mask, device):
    """
    Evaluate model

    Args:
        model: Diffusion model
        data: PyG data object
        mask: Evaluation mask
        device: Computing device

    Returns:
        float: Loss value
        float: Accuracy score
    """
    model.eval()

    # Move data to device
    x = data.x.to(device)
    y = data.y.to(device)
    mask = mask.to(device)

    # Forward pass
    logits = model(x, training=False)

    # Calculate metrics
    loss = F.cross_entropy(logits[mask], y[mask])
    acc = accuracy(logits[mask], y[mask])

    return loss.item(), acc