"""Implementation of the kernelized Gumbel-Softmax operator and related network components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KernelizedGumbelSoftmax(nn.Module):
    """Kernelized Gumbel-Softmax operator for differentiable graph structure learning."""

    def __init__(self, temperature=0.4, hard=False):
        """Initialize the operator.

        Args:
            temperature (float): Temperature parameter for Gumbel-Softmax
            hard (bool): Whether to use hard sampling
        """
        super().__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(self, logits, adjacency_matrix=None, relational_bias=True):
        """Forward pass of the kernelized Gumbel-Softmax operator.

        Args:
            logits (torch.Tensor): Input logits for edge probabilities [N, N]
            adjacency_matrix (torch.Tensor, optional): Existing graph structure [N, N]
            relational_bias (bool): Whether to apply relational bias

        Returns:
            torch.Tensor: Differentiable graph structure [N, N]
        """
        device = logits.device
        batch_size = logits.size(0)

        if logits.is_cuda:
            gumbel = torch.distributions.Gumbel(
                torch.zeros_like(logits),
                torch.ones_like(logits)
            )
        else:
            gumbel = torch.distributions.Gumbel(
                torch.zeros_like(logits),
                torch.ones_like(logits)
            )

        # Apply relational bias if specified and adjacency matrix is provided
        if relational_bias and adjacency_matrix is not None:
            if adjacency_matrix.size() != logits.size():
                raise ValueError(f"Adjacency matrix size {adjacency_matrix.size()} does not match logits size {logits.size()}")
            edge_weights = F.sigmoid(adjacency_matrix)
            logits = logits * edge_weights

        # Sample with Gumbel noise
        if self.hard:
            # Hard sampling (one-hot)
            z = torch.argmax(logits + gumbel.sample(), dim=-1)
            y_hard = F.one_hot(z, num_classes=logits.size(-1)).float()
            # Straight-through gradient estimator
            y = y_hard - logits.detach() + logits
        else:
            # Soft sampling
            gumbel_softmax_sample = F.softmax((logits + gumbel.sample()) / self.temperature, dim=-1)
            y = gumbel_softmax_sample

        return y