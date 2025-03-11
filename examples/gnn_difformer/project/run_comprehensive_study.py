import torch
import numpy as np
import json
import os
from model.improved_diffusion import ImprovedDiffusionModel
from data_processing.dataset import load_dataset, get_train_val_test_split
from training.train import train_epoch, evaluate
import logging

def setup_experiment_logging(log_dir='./logs/comprehensive_study'):
    """Setup logging for comprehensive experiments"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'experiment.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(), log_dir

def train_and_evaluate(model, data, train_mask, val_mask, test_mask, config, device, logger):
    """Train and evaluate model with detailed tracking"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    
    results = {
        'train_acc': [], 'train_loss': [],
        'val_acc': [], 'val_loss': [],
        'best_val_acc': 0, 'best_epoch': 0,
        'test_acc': 0
    }
    
    try:
        for epoch in range(config['epochs']):
            # Training
            train_loss, train_acc = train_epoch(model, data, optimizer, device, epoch)
            
            # Validation
            val_loss, val_acc = evaluate(model, data, val_mask, device)
            
            # Learning rate adjustment
            scheduler.step()
            
            # Record metrics
            results['train_acc'].append(train_acc)
            results['train_loss'].append(train_loss)
            results['val_acc'].append(val_acc)
            results['val_loss'].append(val_loss)
            
            # Track best performance
            if val_acc > results['best_val_acc']:
                results['best_val_acc'] = val_acc
                results['best_epoch'] = epoch
                # Test performance at best validation
                test_loss, test_acc = evaluate(model, data, test_mask, device)
                results['test_acc'] = test_acc
            
            # Log every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{config['epochs']}: "
                    f"Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}"
                )
    
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return None
    
    return results

def run_hyperparam_study(data, train_mask, val_mask, test_mask, device, logger):
    """Run hyperparameter study"""
    # Hyperparameter configurations
    configs = [
        {'name': 'baseline', 'hidden_dim': 64, 'num_layers': 2, 'tau': 0.1, 
         'lambda_reg': 1.0, 'dropout': 0.1, 'learning_rate': 0.001, 'epochs': 100},
        {'name': 'high_reg', 'hidden_dim': 64, 'num_layers': 2, 'tau': 0.1, 
         'lambda_reg': 2.0, 'dropout': 0.1, 'learning_rate': 0.001, 'epochs': 100},
        {'name': 'deep_model', 'hidden_dim': 64, 'num_layers': 4, 'tau': 0.1, 
         'lambda_reg': 1.0, 'dropout': 0.1, 'learning_rate': 0.001, 'epochs': 100},
        {'name': 'high_dropout', 'hidden_dim': 64, 'num_layers': 2, 'tau': 0.1, 
         'lambda_reg': 1.0, 'dropout': 0.3, 'learning_rate': 0.001, 'epochs': 100},
        {'name': 'small_tau', 'hidden_dim': 64, 'num_layers': 2, 'tau': 0.05, 
         'lambda_reg': 1.0, 'dropout': 0.1, 'learning_rate': 0.001, 'epochs': 100}
    ]
    
    results = {}
    num_features = data.x.size(1)
    num_classes = data.y.max().item() + 1
    
    for config in configs:
        logger.info(f"\nRunning configuration: {config['name']}")
        
        model = ImprovedDiffusionModel(
            input_dim=num_features,
            hidden_dim=config['hidden_dim'],
            num_classes=num_classes,
            num_layers=config['num_layers'],
            tau=config['tau'],
            lambda_reg=config['lambda_reg'],
            dropout=config['dropout']
        ).to(device)
        
        experiment_results = train_and_evaluate(
            model, data, train_mask, val_mask, test_mask, config, device, logger
        )
        
        if experiment_results is not None:
            results[config['name']] = experiment_results
            logger.info(
                f"Configuration {config['name']} completed:\n"
                f"Best val acc: {experiment_results['best_val_acc']:.4f} "
                f"at epoch {experiment_results['best_epoch']}\n"
                f"Test acc: {experiment_results['test_acc']:.4f}"
            )
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
    
    return results

def main():
    # Setup
    logger, log_dir = setup_experiment_logging()
    device = torch.device('cuda:2')  # Use GPU 2
    logger.info(f"Using device: {device}")
    
    # Run experiments on multiple datasets
    datasets = ['Cora', 'CiteSeer']
    comprehensive_results = {}
    
    for dataset_name in datasets:
        logger.info(f"\nProcessing dataset: {dataset_name}")
        
        try:
            # Load dataset
            data, num_features, num_classes = load_dataset(dataset_name)
            train_mask, val_mask, test_mask = get_train_val_test_split(data)
            
            # Run hyperparameter study
            results = run_hyperparam_study(
                data, train_mask, val_mask, test_mask, device, logger
            )
            
            comprehensive_results[dataset_name] = results
            
            # Save intermediate results
            with open(os.path.join(log_dir, f'{dataset_name}_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {e}")
            continue
    
    # Save final results
    with open(os.path.join(log_dir, 'comprehensive_results.json'), 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    logger.info("Comprehensive study completed. Results saved in " + log_dir)

if __name__ == "__main__":
    main()