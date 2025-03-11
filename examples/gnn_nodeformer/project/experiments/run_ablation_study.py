"""Ablation study experiments for the model components."""

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
    filename=f'ablation_study_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def run_experiment(config, dataset="Cora"):
    """Run a single experiment with given configuration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, (num_features, num_classes) = load_dataset(dataset)
    data = data.to(device)
    
    # Convert edge_index to adjacency matrix
    adj_matrix = edge_index_to_adj(data.edge_index, data.x.size(0))
    
    # Initialize model with config
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
    
    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with current configuration
        logits, graph_structure = model(data.x, adj_matrix)
        
        # Compute loss based on ablation settings
        loss, loss_components = combined_loss(
            logits, data.y, graph_structure, adj_matrix,
            mask=data.train_mask,
            edge_weight=config.get("edge_weight", 0.1)
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
                    
                logging.info(f"Epoch {epoch+1}: Loss={loss.item():.4f}, "
                           f"Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}")
    
    return {
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "config": config
    }

def ablation_study():
    """Run comprehensive ablation study."""
    # Base configuration
    base_config = {
        "epochs": 200,
        "hidden_dim": 64,
        "num_layers": 2,
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
        "dropout": 0.5,
        "temperature": 0.4,
        "edge_weight": 0.1
    }
    
    # Ablation configurations
    ablation_configs = {
        "baseline": base_config,
        "no_edge_reg": {**base_config, "edge_weight": 0.0},
        "high_temp": {**base_config, "temperature": 1.0},
        "low_temp": {**base_config, "temperature": 0.1},
        "deep_model": {**base_config, "num_layers": 4},
        "high_dropout": {**base_config, "dropout": 0.8}
    }
    
    results = {}
    for name, config in ablation_configs.items():
        logging.info(f"\nRunning ablation study: {name}")
        results[name] = run_experiment(config)
        
        # Log results
        logging.info(f"Configuration: {name}")
        logging.info(f"Best validation accuracy: {results[name]['best_val_acc']:.4f}")
        logging.info(f"Best test accuracy: {results[name]['best_test_acc']:.4f}")
    
    # Save final results
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    ablation_study()