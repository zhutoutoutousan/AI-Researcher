import os
import torch
import logging
from datetime import datetime
from data_processing.dataset import load_data, create_adj_matrices, RecommenderDataset
from model.contrastive_model import HeteroContrastiveModel
from training.trainer import Trainer

def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        filename=f'logs/ablation_{timestamp}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def run_ablation_experiment(config, ablation_type):
    logging.info(f"Starting ablation experiment: {ablation_type}")
    logging.info(f"Configuration: {config}")
    
    # Data loading
    data_dir = "data/yelp"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_mat, test_mat = load_data(data_dir)
    num_users, num_items = train_mat.shape

    # Graph construction
    user_item_graph, user_graph, item_graph = create_adj_matrices(train_mat)
    user_item_graph = user_item_graph.to(device).float()
    user_graph = user_graph.to(device).float()
    item_graph = item_graph.to(device).float()

    # Model initialization
    dataset = RecommenderDataset(train_mat, batch_size=config['batch_size'])
    model = HeteroContrastiveModel(num_users, num_items, 
                                 config['embed_dim'], 
                                 config['n_layers'],
                                 config['temperature'],
                                 config['lambda_coef'])
    
    if ablation_type == 'no_meta':
        model.meta_net = None  # Disable meta network
    elif ablation_type == 'no_contrastive':
        model.calc_ssl_loss = lambda *args, **kwargs: 0  # Disable contrastive loss
    elif ablation_type == 'no_gnn':
        model.main_gnn.n_layers = 0  # Disable GNN layers
    
    trainer = Trainer(model, dataset, config['learning_rate'], device)

    # Training loop
    best_recall = 0
    best_ndcg = 0
    eval_freq = config['eval_frequency']
    early_stop_counter = 0
    early_stop_patience = config['early_stop_patience']

    for epoch in range(config['num_epochs']):
        loss = trainer.train_epoch(user_graph, user_item_graph, item_graph)
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']} - Loss: {loss:.4f}")

        if (epoch + 1) % eval_freq == 0:
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

            if early_stop_counter >= early_stop_patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Final evaluation
    recall, ndcg = trainer.evaluate(user_graph, user_item_graph, item_graph,
                                  test_mat, k=20)
    logging.info(f"Final Results - Recall@20: {recall:.4f}, NDCG@20: {ndcg:.4f}")
    logging.info(f"Best Results - Recall@20: {best_recall:.4f}, NDCG@20: {best_ndcg:.4f}")

    return {
        'best_recall': best_recall,
        'best_ndcg': best_ndcg,
        'final_recall': recall,
        'final_ndcg': ndcg
    }

def run_all_ablations():
    base_config = {
        'batch_size': 2048,
        'embed_dim': 64,
        'n_layers': 2,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'temperature': 0.1,
        'lambda_coef': 0.5,
        'eval_frequency': 5,
        'early_stop_patience': 10
    }

    # Run different ablation configurations
    ablation_types = ['no_meta', 'no_contrastive', 'no_gnn']
    ablation_results = {}

    for ablation_type in ablation_types:
        logging.info(f"\nRunning ablation experiment: {ablation_type}")
        ablation_results[ablation_type] = run_ablation_experiment(base_config, ablation_type)

    # Log comparative results
    logging.info("\nAblation Study Results Summary:")
    for ablation_type, result in ablation_results.items():
        logging.info(f"\n{ablation_type}:")
        logging.info(f"Best Recall@20: {result['best_recall']:.4f}")
        logging.info(f"Best NDCG@20: {result['best_ndcg']:.4f}")

if __name__ == "__main__":
    setup_logging()
    run_all_ablations()