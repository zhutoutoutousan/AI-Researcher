import torch
import torch.optim as optim
from model.vae import FiniteScalarQuantizedVAE
from data_processing.dataset import get_data_loaders
from training.trainer import Trainer
import json
import os
import logging

# Setup logging
logging.basicConfig(filename='temperature_annealing.log',
                   level=logging.INFO,
                   format='%(asctime)s - %(message)s')

def main():
    # Set device
    device = torch.cuda.current_device()
    logging.info(f"Using device: {device}")

    # Set hyperparameters with temperature annealing
    config = {
        'batch_size': 32,  # Reduced batch size
        'num_workers': 2,
        'latent_dim': 32,  # Reduced latent dimension
        'hidden_dim': 32,  # Reduced hidden dimension
        'num_levels': 10,
        'learning_rate': 1e-3,
        'num_epochs': 50,
        'initial_temperature': 1.0,
        'min_temperature': 0.1,
        'annealing_factor': 0.98
    }

    # Create data loaders
    logging.info("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(
        data_dir='data',
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Initialize model
    logging.info("Initializing model...")
    model = FiniteScalarQuantizedVAE(
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        num_levels=config['num_levels']
    ).to(device)

    # Initialize optimizer with gradient clipping
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Initialize trainer with temperature annealing
    temperature = config['initial_temperature']
    history = {'train_loss': [], 'test_loss': [], 'temperature': [], 'fid': []}

    for epoch in range(config['num_epochs']):
        logging.info(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        logging.info(f"Current temperature: {temperature:.4f}")

        # Update model's temperature
        model.quantizer.set_temperature(temperature)

        # Initialize trainer
        trainer = Trainer(model, train_loader, test_loader, optimizer, device)
        
        # Train epoch
        train_metrics = trainer.train_epoch(epoch)

        # Clear GPU cache
        torch.cuda.empty_cache()
        
        test_metrics = trainer.evaluator.evaluate()

        # Log metrics
        logging.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logging.info(f"Test Loss: {test_metrics['loss']:.4f}")
        logging.info(f"FID Score: {test_metrics['fid']:.4f}")

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['test_loss'].append(test_metrics['loss'])
        history['temperature'].append(temperature)
        history['fid'].append(test_metrics['fid'])

        # Update temperature
        temperature = max(
            temperature * config['annealing_factor'],
            config['min_temperature']
        )

        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint_path = f'checkpoints/temperature_annealing_epoch_{epoch}.pt'
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'temperature': temperature,
                'history': history
            }, checkpoint_path)
            logging.info(f"Saved checkpoint at epoch {epoch}")

        # Clear GPU cache
        torch.cuda.empty_cache()

    # Save final training history
    with open('temperature_annealing_history.json', 'w') as f:
        json.dump(history, f)
    logging.info("Training history saved to temperature_annealing_history.json")

    # Final evaluation
    final_metrics = trainer.evaluator.evaluate()
    logging.info("\nFinal Evaluation:")
    logging.info(f"Test Loss: {final_metrics['loss']:.4f}")
    logging.info(f"FID Score: {final_metrics['fid']:.4f}")

if __name__ == '__main__':
    main()