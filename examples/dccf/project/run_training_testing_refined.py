import numpy as np
import torch
import torch.utils.data as data
from model.intentgcl import IntentGCL
from model.utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
from data_processing.dataset import load_data, TrnData
from tqdm import tqdm
import pandas as pd
import time
import os
import logging

# Set up logging
logging.basicConfig(
    filename=f'/workplace/project/training_logs_{time.strftime("%Y-%m-%d-%H-%M")}.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

class Config:
    def __init__(self, **kwargs):
        self.embed_dim = kwargs.get('embed_dim', 64)
        self.n_layers = kwargs.get('n_layers', 2)
        self.n_intents = kwargs.get('n_intents', 128)
        self.temp = kwargs.get('temp', 0.2)
        self.lambda_1 = kwargs.get('lambda_1', 0.2)  # BPR loss weight
        self.lambda_2 = kwargs.get('lambda_2', 1.0)  # Contrastive loss weight
        self.dropout = kwargs.get('dropout', 0.1)
        self.batch_size = kwargs.get('batch_size', 2048)
        self.inter_batch = kwargs.get('inter_batch', 4096)
        self.lr = kwargs.get('lr', 1e-3)
        self.epochs = kwargs.get('epochs', 50)
        self.device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

def train_and_evaluate(config, experiment_name):
    logging.info(f"Starting experiment: {experiment_name}")
    logging.info(f"Configuration: {vars(config)}")

    # Load dataset
    train, train_csr, test_labels = load_data('/workplace/project/data/gowalla/')
    train_data = TrnData(train)
    train_loader = data.DataLoader(train_data, batch_size=config.inter_batch, shuffle=True, num_workers=0)

    # Create normalized adjacency matrix
    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
    adj_norm = adj_norm.coalesce().to(config.device)

    # SVD enhancement
    adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().to(config.device)
    svd_u, s, svd_v = torch.svd_lowrank(adj, q=5)
    u_mul_s = svd_u @ torch.diag(s)
    v_mul_s = svd_v @ torch.diag(s)

    # Initialize model
    model = IntentGCL(
        n_users=train.shape[0],
        n_items=train.shape[1],
        embed_dim=config.embed_dim,
        u_mul_s=u_mul_s,
        v_mul_s=v_mul_s,
        ut=svd_u.T,
        vt=svd_v.T,
        train_csr=train_csr,
        adj_norm=adj_norm,
        n_layers=config.n_layers,
        temp=config.temp,
        lambda_1=config.lambda_1,
        lambda_2=config.lambda_2,
        dropout=config.dropout,
        n_intents=config.n_intents,
        batch_user=config.batch_size,
        device=config.device
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    metrics_history = {
        'epoch': [], 'loss': [], 'loss_bpr': [], 'loss_contrast': [],
        'recall@20': [], 'ndcg@20': [], 'recall@40': [], 'ndcg@40': [],
        'mad': []  # Mean Average Distance for over-smoothing monitoring
    }

    best_recall20 = 0
    best_epoch = 0

    for epoch in range(config.epochs):
        # Training
        model.train()
        epoch_loss = epoch_loss_bpr = epoch_loss_contrast = 0
        train_loader.dataset.neg_sampling()

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}'):
            user_ids, pos_items, neg_items = [x.to(config.device) for x in batch]
            item_ids = torch.cat([pos_items, neg_items], dim=0)

            optimizer.zero_grad()
            loss, loss_bpr, loss_contrast = model(user_ids, item_ids, pos_items, neg_items)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_bpr += loss_bpr.item()
            epoch_loss_contrast += loss_contrast.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_loss_bpr = epoch_loss_bpr / len(train_loader)
        avg_loss_contrast = epoch_loss_contrast / len(train_loader)

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_users = torch.arange(train.shape[0]).to(config.device)
            all_recall_20 = all_ndcg_20 = all_recall_40 = all_ndcg_40 = 0
            all_embeddings = []
            batch_count = (len(test_users) + config.batch_size - 1) // config.batch_size

            for i in range(batch_count):
                start = i * config.batch_size
                end = min((i + 1) * config.batch_size, len(test_users))
                batch_users = test_users[start:end]

                predictions = model(batch_users, None, None, None, test=True)
                predictions = predictions.cpu().numpy()

                recall_20, ndcg_20 = metrics(test_users[start:end].cpu().numpy(), predictions, 20, test_labels)
                recall_40, ndcg_40 = metrics(test_users[start:end].cpu().numpy(), predictions, 40, test_labels)

                all_recall_20 += recall_20
                all_ndcg_20 += ndcg_20
                all_recall_40 += recall_40
                all_ndcg_40 += ndcg_40

                all_embeddings.append(predictions)

            # Calculate metrics
            avg_recall_20 = all_recall_20 / batch_count
            avg_ndcg_20 = all_ndcg_20 / batch_count
            avg_recall_40 = all_recall_40 / batch_count
            avg_ndcg_40 = all_ndcg_40 / batch_count

            # Calculate MAD (Mean Average Distance) for over-smoothing monitoring
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            mad = np.mean(np.abs(all_embeddings - np.mean(all_embeddings, axis=0)))

            # Log metrics
            log_msg = (f'Epoch {epoch+1}: Loss={avg_loss:.4f}, BPR Loss={avg_loss_bpr:.4f}, '
                      f'Contrast Loss={avg_loss_contrast:.4f}\n'
                      f'Recall@20={avg_recall_20:.4f}, NDCG@20={avg_ndcg_20:.4f}, '
                      f'Recall@40={avg_recall_40:.4f}, NDCG@40={avg_ndcg_40:.4f}, '
                      f'MAD={mad:.4f}')
            logging.info(log_msg)

            # Store metrics
            for key, value in zip(
                metrics_history.keys(),
                [epoch+1, avg_loss, avg_loss_bpr, avg_loss_contrast,
                 avg_recall_20, avg_ndcg_20, avg_recall_40, avg_ndcg_40, mad]
            ):
                metrics_history[key].append(value)

            # Update learning rate
            scheduler.step(avg_recall_20)

            # Save best model
            if avg_recall_20 > best_recall20:
                best_recall20 = avg_recall_20
                best_epoch = epoch
                torch.save(model.state_dict(), f'/workplace/project/best_model_{experiment_name}.pt')

    # Save results
    results_df = pd.DataFrame(metrics_history)
    results_df.to_csv(f'/workplace/project/results_{experiment_name}.csv', index=False)
    logging.info(f"Best model at epoch {best_epoch+1} with Recall@20={best_recall20:.4f}")

    return metrics_history

def main():
    # Experiment configurations
    experiments = {
        'base': {},
        'deeper_gnn': {'n_layers': 3, 'dropout': 0.2},
        'more_intents': {'n_intents': 256, 'lambda_2': 1.5},
        'balanced_loss': {'lambda_1': 1.0, 'lambda_2': 1.0, 'temp': 0.1},
    }

    results = {}
    for name, params in experiments.items():
        try:
            logging.info(f"\n{'='*50}\nStarting experiment set: {name}\n{'='*50}")
            config = Config(**params)
            results[name] = train_and_evaluate(config, name)
        except Exception as e:
            logging.error(f"Error in experiment {name}: {str(e)}")
            continue

    # Save comparative results
    comparative_metrics = {}
    for name, history in results.items():
        idx = history['recall@20'].index(max(history['recall@20']))
        comparative_metrics[name] = {
            'best_recall@20': history['recall@20'][idx],
            'best_ndcg@20': history['ndcg@20'][idx],
            'best_recall@40': history['recall@40'][idx],
            'best_ndcg@40': history['ndcg@40'][idx],
            'final_mad': history['mad'][idx],
            'best_epoch': idx + 1
        }

    pd.DataFrame(comparative_metrics).to_csv('/workplace/project/comparative_results.csv')
    logging.info("\nExperiments completed. Results saved in comparative_results.csv")

if __name__ == "__main__":
    main()