import os
import torch
import logging
import numpy as np
from datetime import datetime
from data_processing.dataset import load_data, create_adj_matrices, RecommenderDataset
from model.contrastive_model import HeteroContrastiveModel
from training.trainer import Trainer

def setup_logging():
    if not os.path.exists('logs/final_experiments'):
        os.makedirs('logs/final_experiments')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        filename=f'logs/final_experiments/experiment_{timestamp}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def run_experiments():
    base_config = {
        'batch_size': 512,
        'embed_dim': 32,  # Using 32 as per analysis
        'n_layers': 1,    # Single layer as per analysis
        'learning_rate': 0.001,
        'num_epochs': 35, # Early stopping around 35 epochs
        'temperature': 0.5,  # Best temperature from analysis
        'lambda_coef': 0.5,
        'eval_frequency': 5,
        'early_stop_patience': 10,
    }

    # Test on multiple datasets
    datasets = ['yelp', 'gowalla']  # Add more datasets later
    results = {}

    for dataset in datasets:
        logging.info(f"\nStarting experiments on {dataset} dataset")
        data_dir = f"data/{dataset}"
        
        try:
            # Device configuration
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
            
            # Data loading
            train_mat, test_mat = load_data(data_dir)
            num_users, num_items = train_mat.shape
            logging.info(f"{dataset} dataset loaded: {num_users} users, {num_items} items")

            # Graph construction
            user_item_graph, user_graph, item_graph = create_adj_matrices(train_mat)
            user_item_graph = user_item_graph.to(device).float()
            user_graph = user_graph.to(device).float()
            item_graph = item_graph.to(device).float()

            # Model initialization
            dataset_obj = RecommenderDataset(train_mat, batch_size=base_config['batch_size'])
            model = HeteroContrastiveModel(
                num_users, num_items,
                base_config['embed_dim'],
                base_config['n_layers'],
                base_config['temperature'],
                base_config['lambda_coef']
            ).to(device)

            trainer = Trainer(model, dataset_obj, base_config['learning_rate'], device)

            best_recall = 0
            best_ndcg = 0
            early_stop_counter = 0
            
            # Training loop
            for epoch in range(base_config['num_epochs']):
                loss = trainer.train_epoch(user_graph, user_item_graph, item_graph)
                logging.info(f"Epoch {epoch+1}/{base_config['num_epochs']} - Loss: {loss:.4f}")

                if (epoch + 1) % base_config['eval_frequency'] == 0:
                    recall, ndcg = trainer.evaluate(user_graph, user_item_graph, item_graph,
                                                test_mat, k=20)
                    logging.info(f"Epoch {epoch+1} Eval - Recall@20: {recall:.4f}, NDCG@20: {ndcg:.4f}")

                    if recall > best_recall:
                        best_recall = recall
                        best_ndcg = ndcg
                        early_stop_counter = 0
                        logging.info(f"New best performance - Recall@20: {recall:.4f}, NDCG@20: {ndcg:.4f}")
                    else:
                        early_stop_counter += 1

                    if early_stop_counter >= base_config['early_stop_patience']:
                        logging.info(f"Early stopping triggered at epoch {epoch+1}")
                        break

            # Final evaluation
            recall, ndcg = trainer.evaluate(user_graph, user_item_graph, item_graph,
                                          test_mat, k=20)
            
            results[dataset] = {
                'best_recall': best_recall,
                'best_ndcg': best_ndcg,
                'final_recall': recall,
                'final_ndcg': ndcg
            }

            logging.info(f"\nFinal Results for {dataset}:")
            logging.info(f"Best - Recall@20: {best_recall:.4f}, NDCG@20: {best_ndcg:.4f}")
            logging.info(f"Final - Recall@20: {recall:.4f}, NDCG@20: {ndcg:.4f}")

        except Exception as e:
            logging.error(f"Error during experiment on {dataset}: {str(e)}")
            continue

        finally:
            try:
                torch.cuda.empty_cache()
            except:
                pass

    # Log comparative results
    logging.info("\nComparative Results Across Datasets:")
    for dataset, metrics in results.items():
        logging.info(f"\n{dataset}:")
        logging.info(f"Best Performance - Recall@20: {metrics['best_recall']:.4f}, NDCG@20: {metrics['best_ndcg']:.4f}")
        logging.info(f"Final Performance - Recall@20: {metrics['final_recall']:.4f}, NDCG@20: {metrics['final_ndcg']:.4f}")

if __name__ == "__main__":
    setup_logging()
    logging.info("Starting final experiments with optimized configuration")
    run_experiments()