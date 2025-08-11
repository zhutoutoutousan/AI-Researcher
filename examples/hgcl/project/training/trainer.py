import torch
import numpy as np
import torch.optim as optim
from testing.metrics import recall_and_ndcg_at_k

class Trainer:
    def __init__(self, model, dataset, learning_rate=0.001, device='cuda'):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Move model to device
        self.model = self.model.to(device)
        
    def train_epoch(self, user_graph, user_item_graph, item_graph):
        self.model.train()
        total_loss = 0
        n_batch = 0
        
        while n_batch * self.dataset.batch_size < self.dataset.num_users:
            users, pos_items, neg_items = self.dataset.sample()
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)
            
            self.optimizer.zero_grad()
            loss = self.model.calculate_loss(user_graph, user_item_graph, item_graph,
                                           users, pos_items, neg_items)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batch += 1
            
        return total_loss / n_batch
        
    @torch.no_grad()
    def evaluate(self, user_graph, user_item_graph, item_graph, test_mat, k=20):
        self.model.eval()
        
        # Get embeddings
        user_emb, item_emb, _, _ = self.model(user_graph, user_item_graph, item_graph)
        
        # Calculate predictions
        predictions = torch.mm(user_emb, item_emb.t())
        predictions = predictions.cpu().numpy()
        
        # Get test labels
        test_labels = {}
        for (u, i) in zip(*test_mat.nonzero()):
            if u not in test_labels:
                test_labels[u] = []
            test_labels[u].append(i)
            
        # Calculate metrics
        recall, ndcg = recall_and_ndcg_at_k(predictions, test_labels, k)
        
        return recall, ndcg