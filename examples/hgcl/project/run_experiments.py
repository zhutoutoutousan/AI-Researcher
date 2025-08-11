import os
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from data_processing.dataset import load_data, create_adj_matrices, RecommenderDataset
from model.contrastive_model import HeteroContrastiveModel
from training.trainer import Trainer

def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        filename=f'logs/experiment_{timestamp}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def visualize_results(data_dict, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(10, 6))
    x_values = list(data_dict.keys())
    recall_values = [data_dict[x]['best_recall'] for x in x_values]
    ndcg_values = [data_dict[x]['best_ndcg'] for x in x_values]

    plt.plot(x_values, recall_values, 'o-', label='Recall@20')
    plt.plot(x_values, ndcg_values, 's-', label='NDCG@20')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def run_experiment(config):
    logging.info(f"Starting experiment with config: {config}")
    
    # Configuration
    data_dir = "data/yelp"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()

    # Load data
    train_mat, test_mat = load_data(data_dir)
    num_users, num_items = train_mat.shape
    logging.info(f"Dataset loaded: {num_users} users, {num_items} items")

    # Create adjacency matrices
    user_item_graph, user_graph, item_graph = create_adj_matrices(train_mat)
    user_item_graph = user_item_graph.to(device).float()
    user_graph = user_graph.to(device).float()
    item_graph = item_graph.to(device).float()

    # Initialize dataset and model
    dataset = RecommenderDataset(train_mat, batch_size=config['batch_size'])
    model = HeteroContrastiveModel(num_users, num_items, 
                                 config['embed_dim'], 
                                 config['n_layers'],
                                 config['temperature'],
                                 config['lambda_coef']).to(device)
    trainer = Trainer(model, dataset, config['learning_rate'], device)

    try:
        results_history = {'epoch': [], 'loss': [], 'recall': [], 'ndcg': []}
        best_recall = 0
        best_ndcg = 0
        early_stop_counter = 0
        
        for epoch in range(config['num_epochs']):
            loss = trainer.train_epoch(user_graph, user_item_graph, item_graph)
            logging.info(f"Epoch {epoch+1}/{config['num_epochs']} - Loss: {loss:.4f}")
            
            results_history['epoch'].append(epoch + 1)
            results_history['loss'].append(loss)

            if (epoch + 1) % config['eval_frequency'] == 0:
                recall, ndcg = trainer.evaluate(user_graph, user_item_graph, item_graph,
                                             test_mat, k=20)
                logging.info(f"Epoch {epoch+1} Eval - Recall@20: {recall:.4f}, NDCG@20: {ndcg:.4f}")
                
                results_history['recall'].append(recall)
                results_history['ndcg'].append(ndcg)

                if recall > best_recall:
                    best_recall = recall
                    best_ndcg = ndcg
                    early_stop_counter = 0
                    logging.info(f"New best performance - Recall@20: {recall:.4f}, NDCG@20: {ndcg:.4f}")
                else:
                    early_stop_counter += 1

                if early_stop_counter >= config['early_stop_patience']:
                    logging.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

            torch.cuda.empty_cache()

        # Final evaluation
        recall, ndcg = trainer.evaluate(user_graph, user_item_graph, item_graph,
                                      test_mat, k=20)
        logging.info(f"Final Results - Recall@20: {recall:.4f}, NDCG@20: {ndcg:.4f}")
        logging.info(f"Best Results - Recall@20: {best_recall:.4f}, NDCG@20: {best_ndcg:.4f}")

        # Plot training history
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        # Plot loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(results_history['epoch'], results_history['loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.savefig(f'plots/loss_history_{config["temperature"]}_{config["embed_dim"]}_{config["n_layers"]}.png')
        plt.close()

        # Plot metrics
        plt.figure(figsize=(10, 6))
        eval_epochs = list(range(config['eval_frequency'], len(results_history['recall']) * config['eval_frequency'] + 1, config['eval_frequency']))
        plt.plot(eval_epochs, results_history['recall'], label='Recall@20')
        plt.plot(eval_epochs, results_history['ndcg'], label='NDCG@20')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title('Evaluation Metrics')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots/metrics_history_{config["temperature"]}_{config["embed_dim"]}_{config["n_layers"]}.png')
        plt.close()

        return {
            'best_recall': best_recall,
            'best_ndcg': best_ndcg,
            'final_recall': recall,
            'final_ndcg': ndcg,
            'history': results_history
        }

    except Exception as e:
        logging.error(f"Error during experiment: {str(e)}")
        return None

    finally:
        torch.cuda.empty_cache()

def run_sensitivity_analysis():
    base_config = {
        'batch_size': 512,
        'embed_dim': 64,
        'n_layers': 2,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'temperature': 0.1,
        'lambda_coef': 0.5,
        'eval_frequency': 5,
        'early_stop_patience': 10
    }

    # Temperature sensitivity
    temps = [0.05, 0.1, 0.2, 0.5]
    temp_results = {}
    for temp in temps:
        config = base_config.copy()
        config['temperature'] = temp
        logging.info(f"\nTesting temperature: {temp}")
        result = run_experiment(config)
        if result is not None:
            temp_results[temp] = result
        torch.cuda.empty_cache()

    # Embedding dimension sensitivity
    dims = [32, 64, 128]
    dim_results = {}
    for dim in dims:
        config = base_config.copy()
        config['embed_dim'] = dim
        logging.info(f"\nTesting embedding dimension: {dim}")
        result = run_experiment(config)
        if result is not None:
            dim_results[dim] = result
        torch.cuda.empty_cache()

    # Layer sensitivity
    layers = [1, 2, 3]
    layer_results = {}
    for n_layers in layers:
        config = base_config.copy()
        config['n_layers'] = n_layers
        logging.info(f"\nTesting number of layers: {n_layers}")
        result = run_experiment(config)
        if result is not None:
            layer_results[n_layers] = result
        torch.cuda.empty_cache()

    # Generate visualizations
    if not os.path.exists('plots'):
        os.makedirs('plots')

    visualize_results(temp_results, 'Temperature', 'Metric Value', 
                     'Impact of Temperature', 'plots/temperature_sensitivity.png')
    visualize_results(dim_results, 'Embedding Dimension', 'Metric Value',
                     'Impact of Embedding Dimension', 'plots/dimension_sensitivity.png')
    visualize_results(layer_results, 'Number of Layers', 'Metric Value',
                     'Impact of Layer Count', 'plots/layer_sensitivity.png')

    # Log all results
    logging.info("\nSensitivity Analysis Results Summary:")
    
    logging.info("\nTemperature Sensitivity Results:")
    for temp, result in temp_results.items():
        logging.info(f"Temperature {temp}: Recall@20={result['best_recall']:.4f}, NDCG@20={result['best_ndcg']:.4f}")

    logging.info("\nEmbedding Dimension Sensitivity Results:")
    for dim, result in dim_results.items():
        logging.info(f"Dimension {dim}: Recall@20={result['best_recall']:.4f}, NDCG@20={result['best_ndcg']:.4f}")

    logging.info("\nLayer Number Sensitivity Results:")
    for n_layers, result in layer_results.items():
        logging.info(f"Layers {n_layers}: Recall@20={result['best_recall']:.4f}, NDCG@20={result['best_ndcg']:.4f}")

if __name__ == "__main__":
    setup_logging()
    run_sensitivity_analysis()