import torch
import logging
import os
from model.improved_diffusion import ImprovedDiffusionModel
from data_processing.dataset import load_dataset, get_train_val_test_split
from training.train import train_epoch, evaluate
from torch import optim
import json

def setup_logging(experiment_name):
    log_dir = f'./logs/{experiment_name}'
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/experiment.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def run_experiment(config, experiment_name, logger):
    """Run a single experiment with given configuration"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running experiment: {experiment_name} on {device}")
    logger.info(f"Configuration: {config}")

    # Load dataset
    data, num_features, num_classes = load_dataset(name="Cora")
    train_mask, val_mask, test_mask = get_train_val_test_split(data)

    # Initialize model with config
    model = ImprovedDiffusionModel(
        input_dim=num_features,
        hidden_dim=config['hidden_dim'],
        num_classes=num_classes,
        num_layers=config['num_layers'],
        tau=config['tau'],
        lambda_reg=config['lambda_reg'],
        dropout=config['dropout']
    ).to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-6
    )

    # Training loop with early stopping
    best_val_acc = 0
    patience = config['patience']
    patience_counter = 0
    
    results = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(config['epochs']):
        # Training
        train_loss, train_acc = train_epoch(model, data, optimizer, device, epoch=epoch)
        
        # Validation
        val_loss, val_acc = evaluate(model, data, val_mask, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{config['epochs']}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        
        # Store results
        results['train_acc'].append(train_acc)
        results['val_acc'].append(val_acc)
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'./logs/{experiment_name}/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Final testing
    model.load_state_dict(torch.load(f'./logs/{experiment_name}/best_model.pt'))
    test_loss, test_acc = evaluate(model, data, test_mask, device)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Save results
    results['test_acc'] = test_acc
    results['test_loss'] = test_loss
    results['best_val_acc'] = best_val_acc
    
    with open(f'./logs/{experiment_name}/results.json', 'w') as f:
        json.dump(results, f)

    return results

def main():
    # Base configuration
    base_config = {
        'hidden_dim': 128,
        'num_layers': 2,
        'tau': 0.1,
        'lambda_reg': 1.0,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'epochs': 200,
        'patience': 20
    }

    # Define ablation experiments
    experiments = {
        'baseline': base_config.copy(),
        'deeper_model': {**base_config, 'num_layers': 4},
        'stronger_reg': {**base_config, 'lambda_reg': 2.0},
        'lower_tau': {**base_config, 'tau': 0.05},
        'higher_dropout': {**base_config, 'dropout': 0.3},
        'wider_model': {**base_config, 'hidden_dim': 256}
    }

    # Create logs directory
    os.makedirs('./logs', exist_ok=True)

    # Run all experiments
    aggregated_results = {}
    for name, config in experiments.items():
        logger = setup_logging(name)
        logger.info(f"Starting ablation experiment: {name}")
        results = run_experiment(config, name, logger)
        aggregated_results[name] = {
            'test_acc': results['test_acc'],
            'best_val_acc': results['best_val_acc']
        }

    # Save aggregated results
    with open('./logs/ablation_results.json', 'w') as f:
        json.dump(aggregated_results, f)

if __name__ == "__main__":
    main()