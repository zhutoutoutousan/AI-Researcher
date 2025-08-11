import os
import torch
from data_processing.dataset import load_data, create_adj_matrices, RecommenderDataset
from model.contrastive_model import HeteroContrastiveModel
from training.trainer import Trainer

def main():
    # Configuration
    data_dir = "data/yelp"
    embed_dim = 64
    n_layers = 2
    batch_size = 2048
    learning_rate = 0.001
    num_epochs = 50  # Modified to 50 epochs for better convergence
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation_k = 20

    print("Loading data...")
    train_mat, test_mat = load_data(data_dir)
    num_users, num_items = train_mat.shape
    print(f"Dataset loaded: {num_users} users, {num_items} items")

    print("Creating adjacency matrices...")
    user_item_graph, user_graph, item_graph = create_adj_matrices(train_mat)

    # Move graphs to device and convert to float32
    user_item_graph = user_item_graph.to(device).float()
    user_graph = user_graph.to(device).float()
    item_graph = item_graph.to(device).float()

    print("Initializing dataset and model...")
    dataset = RecommenderDataset(train_mat, batch_size=batch_size)
    model = HeteroContrastiveModel(num_users, num_items, embed_dim, n_layers)
    trainer = Trainer(model, dataset, learning_rate, device)

    print("Starting training...")
    best_recall = 0
    best_ndcg = 0

    for epoch in range(num_epochs):
        # Train
        loss = trainer.train_epoch(user_graph, user_item_graph, item_graph)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}")

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            recall, ndcg = trainer.evaluate(user_graph, user_item_graph, item_graph,
                                         test_mat, k=evaluation_k)
            print(f"Epoch {epoch+1} Evaluation - Recall@{evaluation_k}: {recall:.4f}, NDCG@{evaluation_k}: {ndcg:.4f}")
            
            if recall > best_recall:
                best_recall = recall
                best_ndcg = ndcg
                print(f"New best performance - Recall@{evaluation_k}: {recall:.4f}, NDCG@{evaluation_k}: {ndcg:.4f}")

    # Final evaluation
    print("\nFinal evaluation...")
    recall, ndcg = trainer.evaluate(user_graph, user_item_graph, item_graph,
                                  test_mat, k=evaluation_k)
    print(f"Test Results - Recall@{evaluation_k}: {recall:.4f}, NDCG@{evaluation_k}: {ndcg:.4f}")
    print(f"Best Results - Recall@{evaluation_k}: {best_recall:.4f}, NDCG@{evaluation_k}: {best_ndcg:.4f}")

if __name__ == "__main__":
    main()