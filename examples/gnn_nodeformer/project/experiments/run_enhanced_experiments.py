"""Enhanced experiments with improved model architecture."""

import os
import torch
import torch.optim as optim
import logging
import json
from datetime import datetime
import numpy as np

from data_processing.data_loader import load_dataset
from model.enhanced_model import EnhancedGraphLearner
from training.loss import combined_loss
from testing.metrics import accuracy
from experiments.edge_index_to_adj import edge_index_to_adj

# Configure logging
logging.basicConfig(
    filename=f'enhanced_experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def cosine_temperature_scheduler(epoch, max_epochs, t_max=1.0, t_min=0.1):
    """Cosine annealing temperature scheduler."""
    progress = epoch / max_epochs
    temperature = t_min + 0.5 * (t_max - t_min) * (1 + np.cos(np.pi * progress))
    return temperature

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, max_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.max_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

def run_experiment(config, dataset_name="Cora", device_id=0):
    """Run a single experiment with enhanced model."""
    if isinstance(device_id, str) and device_id == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    
    data, (num_features, num_classes) = load_dataset(dataset_name)
    data = data.to(device)
    
    # Adjust num_heads to be compatible with input dimensions
    adjusted_heads = max(1, num_features // 16)  # Ensure it divides input_dim
    config["num_heads"] = adjusted_heads
    
    logging.info(f"Adjusted number of attention heads to {adjusted_heads} for {num_features} features")
    
    # Convert edge_index to adjacency matrix
    adj_matrix = edge_index_to_adj(data.edge_index, data.x.size(0))
    
    # Initialize enhanced model
    model = EnhancedGraphLearner(
        input_dim=num_features,
        hidden_dim=config["hidden_dim"],
        num_classes=num_classes,
        num_layers=config["num_layers"],
        temperature=config["initial_temp"],
        dropout=config["dropout"],
        num_heads=config["num_heads"]
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Setup schedulers
    lr_scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=20,
        total_epochs=config["epochs"],
        max_lr=config["learning_rate"],
        min_lr=1e-6
    )
    
    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    early_stopping_counter = 0
    training_stats = []
    
    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()
        
        try:
            # Update temperature
            current_temp = cosine_temperature_scheduler(
                epoch, config["epochs"],
                t_max=config["initial_temp"],
                t_min=config["min_temp"]
            )
            model.graph_learner.temperature = current_temp
            
            # Forward pass
            logits, graph_structure = model(data.x, adj_matrix)
            
            # Compute loss
            loss, loss_components = combined_loss(
                logits, data.y, graph_structure, adj_matrix,
                mask=data.train_mask,
                edge_weight=config["edge_weight"]
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            
            # Update learning rate
            current_lr = lr_scheduler.step(epoch)
            
            # Validation
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    logits, _ = model(data.x, adj_matrix)
                    val_acc = accuracy(logits[data.val_mask], data.y[data.val_mask])
                    test_acc = accuracy(logits[data.test_mask], data.y[data.test_mask])
                    
                    epoch_stats = {
                        "epoch": epoch + 1,
                        "loss": loss.item(),
                        "val_acc": val_acc,
                        "test_acc": test_acc,
                        "temperature": current_temp,
                        "learning_rate": current_lr
                    }
                    training_stats.append(epoch_stats)
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_test_acc = test_acc
                        early_stopping_counter = 0
                        # Save best model
                        torch.save(model.state_dict(), f'best_model_{dataset_name}.pt')
                    else:
                        early_stopping_counter += 1
                    
                    logging.info(
                        f"Epoch {epoch+1}: Loss={loss.item():.4f}, "
                        f"Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}, "
                        f"Temp={current_temp:.4f}, LR={current_lr:.6f}"
                    )
                    
                    if early_stopping_counter >= config["patience"]:
                        logging.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
                        
        except RuntimeError as e:
            logging.error(f"Error during training: {str(e)}")
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise e
    
    return {
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "training_stats": training_stats,
        "final_temperature": current_temp
    }

def run_enhanced_experiments():
    """Run comprehensive enhanced experiments."""
    config = {
        "epochs": 500,
        "hidden_dim": 128,
        "num_layers": 3,
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
        "dropout": 0.5,
        "initial_temp": 1.0,
        "min_temp": 0.1,
        "edge_weight": 0.1,
        "patience": 50,
        "grad_clip": 5.0
    }
    
    datasets = ["Cora", "CiteSeer", "PubMed"]
    results = {}
    
    # Try different GPU devices
    gpu_devices = list(range(torch.cuda.device_count()))
    if not gpu_devices:
        gpu_devices = ["cpu"]
    
    for dataset in datasets:
        logging.info(f"\nRunning enhanced experiments on {dataset}")
        dataset_results = []
        success = False
        
        for run in range(5):  # 5 runs per dataset
            device_id = gpu_devices[run % len(gpu_devices)]
            logging.info(f"Run {run+1}/5 on device {device_id}")
            
            try:
                run_result = run_experiment(config, dataset, device_id)
                dataset_results.append(run_result)
                success = True
            except RuntimeError as e:
                logging.error(f"Error on device {device_id}: {str(e)}")
                if "cuda" in str(e) or "out of memory" in str(e):
                    logging.info("Falling back to CPU")
                    try:
                        run_result = run_experiment(config, dataset, "cpu")
                        dataset_results.append(run_result)
                        success = True
                    except Exception as cpu_e:
                        logging.error(f"CPU fallback also failed: {str(cpu_e)}")
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
        
        if success and dataset_results:
            # Compute statistics
            val_accs = [r["best_val_acc"] for r in dataset_results]
            test_accs = [r["best_test_acc"] for r in dataset_results]
            
            results[dataset] = {
                "mean_val_acc": np.mean(val_accs),
                "std_val_acc": np.std(val_accs),
                "mean_test_acc": np.mean(test_accs),
                "std_test_acc": np.std(test_accs),
                "training_stats": dataset_results[0]["training_stats"]  # Store first run's training stats
            }
            
            logging.info(f"{dataset} Results:")
            logging.info(f"Validation: {results[dataset]['mean_val_acc']:.4f} ± "
                        f"{results[dataset]['std_val_acc']:.4f}")
            logging.info(f"Test: {results[dataset]['mean_test_acc']:.4f} ± "
                        f"{results[dataset]['std_test_acc']:.4f}")
        else:
            logging.error(f"All runs failed for dataset {dataset}")
            results[dataset] = {
                "error": "All runs failed",
                "mean_val_acc": None,
                "std_val_acc": None,
                "mean_test_acc": None,
                "std_test_acc": None
            }
    
    # Save final results
    with open('enhanced_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_enhanced_experiments()