import numpy as np
import torch
import torch.utils.data as data
from model.intentgcl import IntentGCL
from model.utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
from data_processing.dataset import load_data, TrnData
from tqdm import tqdm
import pandas as pd
import time

# Model Configuration
class Config:
    def __init__(self):
        self.embed_dim = 64
        self.n_layers = 2
        self.n_intents = 128
        self.temp = 0.2
        self.lambda_1 = 0.2
        self.lambda_2 = 1e-7
        self.dropout = 0.0
        self.batch_size = 2048
        self.inter_batch = 4096
        self.lr = 1e-3
        self.epochs = 50  # Changed from 2 to 50 for more comprehensive training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Initialize configuration
    config = Config()
    print(f"Running on device: {config.device}")

    # Load dataset
    print("Loading data...")
    train, train_csr, test_labels = load_data('/workplace/project/data/gowalla/')
    print(f"Dataset loaded. Users: {train.shape[0]}, Items: {train.shape[1]}")

    # Initialize data loaders
    train_data = TrnData(train)
    train_loader = data.DataLoader(train_data, batch_size=config.inter_batch, shuffle=True, num_workers=0)

    # Create normalized adjacency matrix
    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
    adj_norm = adj_norm.coalesce().to(config.device)

    # Perform SVD for enhancement
    print("Performing SVD...")
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

    # Metrics storage
    metrics_history = {
        'epoch': [],
        'loss': [],
        'loss_bpr': [],
        'loss_contrast': [],
        'recall@20': [],
        'ndcg@20': [],
        'recall@40': [],
        'ndcg@40': []
    }

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        epoch_loss_bpr = 0
        epoch_loss_contrast = 0

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

        # Calculate average losses
        batch_count = len(train_loader)
        avg_loss = epoch_loss / batch_count
        avg_loss_bpr = epoch_loss_bpr / batch_count
        avg_loss_contrast = epoch_loss_contrast / batch_count

        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, BPR Loss={avg_loss_bpr:.4f}, Contrast Loss={avg_loss_contrast:.4f}')

        # Testing
        model.eval()
        with torch.no_grad():
            test_users = torch.arange(train.shape[0]).to(config.device)
            batch_size = config.batch_size
            all_recall_20 = all_ndcg_20 = all_recall_40 = all_ndcg_40 = 0
            batch_count = (len(test_users) + batch_size - 1) // batch_size

            for i in tqdm(range(batch_count), desc='Testing'):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(test_users))
                batch_users = test_users[start:end]

                predictions = model(batch_users, None, None, None, test=True)
                predictions = predictions.cpu().numpy()

                recall_20, ndcg_20 = metrics(test_users[start:end].cpu().numpy(), predictions, 20, test_labels)
                recall_40, ndcg_40 = metrics(test_users[start:end].cpu().numpy(), predictions, 40, test_labels)

                all_recall_20 += recall_20
                all_ndcg_20 += ndcg_20
                all_recall_40 += recall_40
                all_ndcg_40 += ndcg_40

            # Calculate average metrics
            avg_recall_20 = all_recall_20 / batch_count
            avg_ndcg_20 = all_ndcg_20 / batch_count
            avg_recall_40 = all_recall_40 / batch_count
            avg_ndcg_40 = all_ndcg_40 / batch_count

            print(f'Test Results: Recall@20={avg_recall_20:.4f}, NDCG@20={avg_ndcg_20:.4f}, '
                  f'Recall@40={avg_recall_40:.4f}, NDCG@40={avg_ndcg_40:.4f}')

            # Store metrics
            metrics_history['epoch'].append(epoch + 1)
            metrics_history['loss'].append(avg_loss)
            metrics_history['loss_bpr'].append(avg_loss_bpr)
            metrics_history['loss_contrast'].append(avg_loss_contrast)
            metrics_history['recall@20'].append(avg_recall_20)
            metrics_history['ndcg@20'].append(avg_ndcg_20)
            metrics_history['recall@40'].append(avg_recall_40)
            metrics_history['ndcg@40'].append(avg_ndcg_40)

    # Save results
    results_df = pd.DataFrame(metrics_history)
    timestamp = time.strftime('%Y-%m-%d-%H-%M')
    results_df.to_csv(f'/workplace/project/results_gowalla_{timestamp}.csv', index=False)
    print(f"Results saved to: results_gowalla_{timestamp}.csv")

if __name__ == "__main__":
    main()