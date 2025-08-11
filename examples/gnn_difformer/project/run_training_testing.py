import torch
import logging
from model.diffusion import DiffusionModel
from data_processing.dataset import load_dataset, get_train_val_test_split
from training.train import train_epoch, evaluate
from torch import optim

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load dataset (using Cora as it's lightweight)
    logger.info("Loading dataset...")
    data, num_features, num_classes = load_dataset(name="Cora")
    train_mask, val_mask, test_mask = get_train_val_test_split(data)

    # Initialize model
    logger.info("Initializing model...")
    model = DiffusionModel(
        input_dim=num_features,
        hidden_dim=256,
        num_classes=num_classes,
        num_layers=2,
        tau=0.1,
        lambda_reg=1.0
    ).to(device)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    logger.info("Starting training...")
    best_val_acc = 0
    for epoch in range(100):  # Train for 100 epochs to get better convergence
        # Training
        train_loss, train_acc = train_epoch(model, data, optimizer, device)
        logger.info(f"Epoch {epoch+1}/100:")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validation
        val_loss, val_acc = evaluate(model, data, val_mask, device)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"New best validation accuracy: {best_val_acc:.4f}")

    # Final testing
    logger.info("Evaluating on test set...")
    test_loss, test_acc = evaluate(model, data, test_mask, device)
    logger.info(f"Final Results:")
    logger.info(f"Best Val Acc: {best_val_acc:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()