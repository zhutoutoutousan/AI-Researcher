"""Comparative study experiments across multiple datasets."""

import os
import torch
import torch.optim as optim
import logging
import json
from datetime import datetime

from data_processing.data_loader import load_dataset
from model.graph_model import KernelizedGraphLearner
from training.loss import combined_loss
from testing.metrics import accuracy
from experiments.edge_index_to_adj import edge_index_to_adj

# Configure logging
logging.basicConfig(
    filename=f'comparative_study_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def run_experiment_dataset(config, dataset_name):
    """Run experiment on a specific dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, (num_features, num_classes) = load_dataset(dataset_name)
    data = data.to(device)
    
    # Convert edge_index to adjacency matrix
    adj_matrix = edge_index_to_adj(data.edge_index, data.x.size(0))
    
    # Initialize model
    model = KernelizedGraphLearner(
        input_dim=num_features,
        hidden_dim=config["hidden_dim"],
        num_classes=num_classes,
        temperature=config["temperature"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=config["weight_decay"]
    )
    
    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    early_stopping_counter = 0
    
    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        logits, graph_structure = model(data.x, adj_matrix)
        
        # Compute loss
        loss, loss_components = combined_loss(
            logits, data.y, graph_structure, adj_matrix,
            mask=data.train_mask,
            edge_weight=config["edge_weight"]
        )
        
        loss.backward()
        optimizer.step()
        
        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(data.x, adj_matrix)
                val_acc = accuracy(logits[data.val_mask], data.y[data.val_mask])
                test_acc = accuracy(logits[data.test_mask], data.y[data.test_mask])
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                logging.info(f"{dataset_name} - Epoch {epoch+1}: "
                           f"Loss={loss.item():.4f}, Val Acc={val_acc:.4f}, "
                           f"Test Acc={test_acc:.4f}")
                
                if early_stopping_counter >= config["patience"]:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
    
    return {
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc
    }

def comparative_study():
    """Run comparative study across datasets."""
    # Configuration
    config = {
        "epochs": 200,
        "hidden_dim": 64,
        "num_layers": 2,
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
        "dropout": 0.5,
        "temperature": 0.4,
        "edge_weight": 0.1,
        "patience": 20
    }
    
    # Datasets to test
    datasets = ["Cora", "CiteSeer", "PubMed"]
    
    results = {}
    for dataset in datasets:
        logging.info(f"\nRunning experiments on {dataset}")
        
        # Multiple runs for statistical significance
        dataset_results = []
        for run in range(5):  # 5 runs per dataset
            logging.info(f"Run {run+1}/5")
            run_result = run_experiment_dataset(config, dataset)
            dataset_results.append(run_result)
        
        # Compute mean and std of results
        val_accs = [r["best_val_acc"] for r in dataset_results]
        test_accs = [r["best_test_acc"] for r in dataset_results]
        
        results[dataset] = {
            "mean_val_acc": sum(val_accs) / len(val_accs),
            "std_val_acc": torch.tensor(val_accs).std().item(),
            "mean_test_acc": sum(test_accs) / len(test_accs),
            "std_test_acc": torch.tensor(test_accs).std().item()
        }
        
        logging.info(f"{dataset} Results:")
        logging.info(f"Validation: {results[dataset]['mean_val_acc']:.4f} ± "
                    f"{results[dataset]['std_val_acc']:.4f}")
        logging.info(f"Test: {results[dataset]['mean_test_acc']:.4f} ± "
                    f"{results[dataset]['std_test_acc']:.4f}")
    
    # Save final results
    with open('comparative_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    comparative_study()