"""Evaluation metrics for graph neural networks."""

import torch

def accuracy(y_pred, y_true):
    """Compute accuracy for node classification.
    
    Args:
        y_pred (torch.Tensor): Model predictions
        y_true (torch.Tensor): Ground truth labels
    
    Returns:
        float: Accuracy score
    """
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)