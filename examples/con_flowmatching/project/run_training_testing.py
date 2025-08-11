import os
import torch
import logging
from data_processing.cifar10 import get_data_loaders
from model.network import VelocityNetwork, CNF
from training.trainer import CNFTrainer
from testing.evaluator import FIDScore, evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Configuration
    data_dir = os.path.join('data', 'cifar-10-batches-py')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 512
    n_epochs = 2
    
    # Set up data loaders with reduced num_workers due to memory constraints
    logger.info("Setting up data loaders...")
    train_loader, test_loader = get_data_loaders(data_dir, batch_size=batch_size, num_workers=1)
    
    # Initialize model
    logger.info("Initializing model...")
    velocity_net = VelocityNetwork()
    model = CNF(velocity_net)
    
    # Initialize trainer
    logger.info("Setting up trainer...")
    trainer = CNFTrainer(model, train_loader, test_loader, device)
    
    # Initialize FID calculator
    logger.info("Setting up FID calculator...")
    fid_calculator = FIDScore(device)
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(n_epochs):
        train_loss = trainer.train_epoch(epoch)
        eval_loss = trainer.evaluate()
        
        logger.info(f"Epoch {epoch+1}/{n_epochs}")
        logger.info(f"Training Loss: {train_loss:.4f}")
        logger.info(f"Evaluation Loss: {eval_loss:.4f}")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    metrics = evaluate_model(trainer.ema_model, test_loader, fid_calculator, device)
    logger.info(f"Final FID Score: {metrics['fid']:.4f}")

if __name__ == "__main__":
    main()