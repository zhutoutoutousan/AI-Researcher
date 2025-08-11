import torch
import logging
import os
import json
from model.improved_diffusion import ImprovedDiffusionModel
from data_processing.dataset import load_dataset, get_train_val_test_split
from training.train import train_epoch, evaluate
from torch import optim
from itertools import product

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/hyperparameter_search.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def train_model_with_config(config, data, train_mask, val_mask, test_mask, device, logger):
    """Train a model with given hyperparameter configuration"""
    num_features = data.x.size(1)
    num_classes = data.y.max().item() + 1

    model = ImprovedDiffusionModel(
        input_dim=num_features,
        hidden_dim=config['hidden_dim'],
        num_classes=num_classes,
        num_layers=config['num_layers'],
        tau=config['tau'],
        lambda_reg=config['lambda_reg'],
        dropout=config['dropout']
    ).to(device)

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

    # Training with early stopping
    best_val_acc = 0
    patience = config['patience']
    patience_counter = 0

    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, data, optimizer, device, epoch=epoch)
        val_loss, val_acc = evaluate(model, data, val_mask, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Evaluate on test set
    model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate(model, data, test_mask, device)

    return {
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'epochs_trained': epoch + 1
    }

def main():
    # Setup
    log_dir = './logs/hyperparameter_search'
    logger = setup_logging(log_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    data, num_features, num_classes = load_dataset(name="Cora")
    train_mask, val_mask, test_mask = get_train_val_test_split(data)

    # Define hyperparameter search space
    param_grid = {
        'hidden_dim': [64, 128, 256],
        'num_layers': [2, 3, 4],
        'tau': [0.05, 0.1, 0.2],
        'lambda_reg': [0.5, 1.0, 2.0],
        'dropout': [0.1, 0.3, 0.5],
        'learning_rate': [0.001, 0.003],
        'weight_decay': [1e-5, 1e-4],
        'epochs': [200],
        'patience': [20]
    }

    # Generate all possible combinations
    keys, values = zip(*param_grid.items())
    configurations = [dict(zip(keys, v)) for v in product(*values)]

    logger.info(f"Total configurations to test: {len(configurations)}")

    # Run grid search
    results = []
    for i, config in enumerate(configurations):
        logger.info(f"Testing configuration {i+1}/{len(configurations)}")
        logger.info(f"Current config: {config}")

        try:
            metrics = train_model_with_config(
                config, data, train_mask, val_mask, test_mask, device, logger
            )
            
            result = {
                'config': config,
                'metrics': metrics
            }
            results.append(result)

            logger.info(f"Results for config {i+1}:")
            logger.info(f"Val Acc: {metrics['val_acc']:.4f}")
            logger.info(f"Test Acc: {metrics['test_acc']:.4f}")
            logger.info(f"Epochs trained: {metrics['epochs_trained']}")

            # Save intermediate results
            with open(f'{log_dir}/results.json', 'w') as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            logger.error(f"Error with configuration {i+1}: {str(e)}")
            continue

    # Find best configuration
    best_result = max(results, key=lambda x: x['metrics']['val_acc'])
    logger.info("\nBest configuration found:")
    logger.info(f"Config: {best_result['config']}")
    logger.info(f"Validation accuracy: {best_result['metrics']['val_acc']:.4f}")
    logger.info(f"Test accuracy: {best_result['metrics']['test_acc']:.4f}")

    # Save final results
    with open(f'{log_dir}/best_config.json', 'w') as f:
        json.dump(best_result, f, indent=2)

if __name__ == "__main__":
    main()