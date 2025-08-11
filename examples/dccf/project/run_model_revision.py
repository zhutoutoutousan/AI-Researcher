import numpy as np
import torch
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
from model.intentgcl import IntentGCL
from model.utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
from data_processing.dataset import load_data, TrnData
from tqdm import tqdm
import pandas as pd
import time
import os
import logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    filename=f'/workplace/project/revision_logs_{time.strftime("%Y-%m-%d-%H-%M")}.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

class RevisionConfig:
    def __init__(self, **kwargs):
        self.embed_dim = kwargs.get('embed_dim', 64)
        self.n_layers = kwargs.get('n_layers', 3)
        self.n_intents = kwargs.get('n_intents', 128)
        self.use_residual = kwargs.get('use_residual', True)
        self.temp = kwargs.get('temp', 0.2)
        self.lambda_1 = kwargs.get('lambda_1', 0.5)  # Balanced BPR loss weight
        self.lambda_2 = kwargs.get('lambda_2', 0.5)  # Balanced contrastive loss weight
        self.lambda_3 = kwargs.get('lambda_3', 1e-4)  # L2 regularization
        self.dropout = kwargs.get('dropout', 0.2)
        self.batch_size = kwargs.get('batch_size', 2048)
        self.inter_batch = kwargs.get('inter_batch', 4096)
        self.lr = kwargs.get('lr', 1e-3)
        self.epochs = kwargs.get('epochs', 50)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_embeddings(user_emb, item_emb, epoch, experiment_name):
    combined_emb = np.vstack([user_emb[:1000], item_emb[:1000]])
    tsne = TSNE(n_components=2, random_state=42)
    vis_emb = tsne.fit_transform(combined_emb)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(vis_emb[:1000, 0], vis_emb[:1000, 1], c='blue', label='Users', alpha=0.5)
    plt.scatter(vis_emb[1000:, 0], vis_emb[1000:, 1], c='red', label='Items', alpha=0.5)
    plt.legend()
    plt.title(f'Embedding Visualization - Epoch {epoch}')
    plt.savefig(f'/workplace/project/embedding_viz_{experiment_name}_epoch_{epoch}.png')
    plt.close()

def train_and_evaluate(config, experiment_name):
    logging.info(f"Starting experiment: {experiment_name}")
    logging.info(f"Configuration: {vars(config)}")

    # Load and preprocess data
    train, train_csr, test_labels = load_data('/workplace/project/data/gowalla/')
    train_data = TrnData(train)
    train_loader = data.DataLoader(train_data, batch_size=config.inter_batch, shuffle=True, num_workers=0)

    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().to(config.device)
    adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().to(config.device)
    
    # SVD enhancement
    svd_u, s, svd_v = torch.svd_lowrank(adj, q=8)  # Increased SVD components
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
        lambda_3=config.lambda_3,
        dropout=config.dropout,
        n_intents=config.n_intents,
        batch_user=config.batch_size,
        device=config.device,
        use_residual=config.use_residual
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    metrics_history = {
        'epoch': [], 'loss': [], 'loss_bpr': [], 'loss_contrast': [], 'loss_reg': [],
        'recall@20': [], 'ndcg@20': [], 'recall@40': [], 'ndcg@40': [],
        'mad': [], 'lr': []
    }

    best_recall20 = 0
    best_epoch = 0
    patience = 10
    patience_counter = 0

    for epoch in range(config.epochs):
        # Training
        model.train()
        epoch_loss = epoch_loss_bpr = epoch_loss_contrast = epoch_loss_reg = 0
        train_loader.dataset.neg_sampling()

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}'):
            user_ids, pos_items, neg_items = [x.to(config.device) for x in batch]
            item_ids = torch.cat([pos_items, neg_items], dim=0)

            optimizer.zero_grad()
            loss, loss_bpr, loss_contrast, loss_reg = model(user_ids, item_ids, pos_items, neg_items)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Added gradient clipping
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_bpr += loss_bpr.item()
            epoch_loss_contrast += loss_contrast.item()
            epoch_loss_reg += loss_reg.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_loss_bpr = epoch_loss_bpr / len(train_loader)
        avg_loss_contrast = epoch_loss_contrast / len(train_loader)
        avg_loss_reg = epoch_loss_reg / len(train_loader)

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

            avg_recall_20 = all_recall_20 / batch_count
            avg_ndcg_20 = all_ndcg_20 / batch_count
            avg_recall_40 = all_recall_40 / batch_count
            avg_ndcg_40 = all_ndcg_40 / batch_count

            # MAD calculation and embedding visualization
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            mad = np.mean(np.abs(all_embeddings - np.mean(all_embeddings, axis=0)))

            if epoch % 10 == 0:
                plot_embeddings(model.final_user_embedding.detach().cpu().numpy(),
                              model.final_item_embedding.detach().cpu().numpy(),
                              epoch, experiment_name)

            # Log metrics
            log_msg = (f'Epoch {epoch+1}: Loss={avg_loss:.4f}, BPR Loss={avg_loss_bpr:.4f}, '
                      f'Contrast Loss={avg_loss_contrast:.4f}, Reg Loss={avg_loss_reg:.4f}\n'
                      f'Recall@20={avg_recall_20:.4f}, NDCG@20={avg_ndcg_20:.4f}, '
                      f'Recall@40={avg_recall_40:.4f}, NDCG@40={avg_ndcg_40:.4f}, '
                      f'MAD={mad:.4f}, LR={optimizer.param_groups[0]["lr"]:.6f}')
            logging.info(log_msg)

            # Store metrics
            metrics_history['epoch'].append(epoch+1)
            metrics_history['loss'].append(avg_loss)
            metrics_history['loss_bpr'].append(avg_loss_bpr)
            metrics_history['loss_contrast'].append(avg_loss_contrast)
            metrics_history['loss_reg'].append(avg_loss_reg)
            metrics_history['recall@20'].append(avg_recall_20)
            metrics_history['ndcg@20'].append(avg_ndcg_20)
            metrics_history['recall@40'].append(avg_recall_40)
            metrics_history['ndcg@40'].append(avg_ndcg_40)
            metrics_history['mad'].append(mad)
            metrics_history['lr'].append(optimizer.param_groups[0]['lr'])

            # Learning rate scheduling and early stopping
            scheduler.step(avg_recall_20)
            
            if avg_recall_20 > best_recall20:
                best_recall20 = avg_recall_20
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), f'/workplace/project/best_model_{experiment_name}.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

    # Save experiment results
    results_df = pd.DataFrame(metrics_history)
    results_df.to_csv(f'/workplace/project/results_{experiment_name}.csv', index=False)
    logging.info(f"Best model at epoch {best_epoch+1} with Recall@20={best_recall20:.4f}")

    return metrics_history

def main():
    experiments = {
        'refined_base': {
            'use_residual': True,
            'lambda_1': 0.5,
            'lambda_2': 0.5,
            'lambda_3': 1e-4
        },
        'hierarchical_intent': {
            'n_intents': 256,
            'use_residual': True,
            'lambda_1': 0.5,
            'lambda_2': 0.5,
            'lambda_3': 1e-4
        },
        'deep_gnn': {
            'n_layers': 4,
            'use_residual': True,
            'lambda_1': 0.5,
            'lambda_2': 0.5,
            'lambda_3': 1e-4,
            'dropout': 0.3
        }
    }

    results = {}
    for name, params in experiments.items():
        try:
            logging.info(f"\n{'='*50}\nStarting experiment set: {name}\n{'='*50}")
            config = RevisionConfig(**params)
            results[name] = train_and_evaluate(config, name)
        except Exception as e:
            logging.error(f"Error in experiment {name}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            continue

    # Save comparative results
    comparative_metrics = {}
    for name, history in results.items():
        if history:  # Check if the experiment completed successfully
            idx = history['recall@20'].index(max(history['recall@20']))
            comparative_metrics[name] = {
                'best_recall@20': history['recall@20'][idx],
                'best_ndcg@20': history['ndcg@20'][idx],
                'best_recall@40': history['recall@40'][idx],
                'best_ndcg@40': history['ndcg@40'][idx],
                'final_mad': history['mad'][idx],
                'best_epoch': idx + 1
            }

    if comparative_metrics:
        pd.DataFrame(comparative_metrics).to_csv('/workplace/project/comparative_results_revision.csv')
    logging.info("\nRevision experiments completed. Results saved.")

if __name__ == "__main__":
    main()