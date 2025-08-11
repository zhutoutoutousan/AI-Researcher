"""Main script for training and testing the graph learning model."""

import os
import torch
import torch.optim as optim

from data_processing.data_loader import load_dataset
from model.graph_model import KernelizedGraphLearner
from training.loss import combined_loss
from testing.metrics import accuracy

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset (using Cora as an example)
    data, num_features, num_classes = load_dataset("Cora")
    data = data.to(device)

    # Initialize model
    model = KernelizedGraphLearner(
        input_dim=num_features,
        hidden_dim=64,
        num_classes=num_classes,
        num_layers=2,
        temperature=0.4,
        dropout=0.5
    ).to(device)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Convert adjacency matrix to dense format for loss computation
    adj_matrix = torch.zeros((data.x.size(0), data.x.size(0)), device=device)
    edge_index = data.edge_index
    adj_matrix[edge_index[0], edge_index[1]] = 1

    # Training loop
    model.train()
    for epoch in range(200):  # Train for 200 epochs to get stable results
        optimizer.zero_grad()

        # Forward pass
        logits, graph_structure = model(data.x, adj_matrix)

        # Compute loss
        loss, loss_components = combined_loss(
            logits, data.y, graph_structure, adj_matrix,
            mask=data.train_mask, edge_weight=0.1
        )

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Compute training accuracy
            train_acc = accuracy(logits[data.train_mask], data.y[data.train_mask])

            print(f"Epoch [{epoch+1}/200]")
            print(f"Loss: {loss_components['total']:.4f} "
                  f"(Classification: {loss_components['classification']:.4f}, "
                  f"Edge Regularization: {loss_components['edge_regularization']:.4f})")
            print(f"Training Accuracy: {train_acc:.4f}")

    # Testing
    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x, adj_matrix)
        test_acc = accuracy(logits[data.test_mask], data.y[data.test_mask])
        print(f"\nTest Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()