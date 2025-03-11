import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HeterogeneousGNN(nn.Module):
    def __init__(self, user_num, item_num, embed_dim, n_layers=2):
        super(HeterogeneousGNN, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_nodes = user_num + item_num

        # Xavier initialization
        self.user_embedding = nn.Parameter(torch.empty(user_num, embed_dim))
        self.item_embedding = nn.Parameter(torch.empty(item_num, embed_dim))
        nn.init.xavier_uniform_(self.user_embedding)
        nn.init.xavier_uniform_(self.item_embedding)

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding
        item_embeddings = self.item_embedding
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, user_graph, user_item_graph, item_graph):
        ego_embeddings = self.get_ego_embeddings()
        all_embeddings = [ego_embeddings]

        user_embeddings = ego_embeddings[:self.user_num]
        item_embeddings = ego_embeddings[self.user_num:]

        for layer in range(self.n_layers):
            # User-Item interaction embeddings
            user_item_embeddings_user = torch.sparse.mm(user_item_graph, item_embeddings)
            user_item_embeddings_item = torch.sparse.mm(user_item_graph.t(), user_embeddings)
            
            # User-User interaction embeddings 
            user_user_embeddings = torch.sparse.mm(user_graph, user_embeddings)
            # Item-Item interaction embeddings
            item_item_embeddings = torch.sparse.mm(item_graph, item_embeddings)
            
            # Combine embeddings for users
            user_combined = torch.stack([
                user_user_embeddings,
                user_item_embeddings_user
            ], dim=1)
            user_embeddings = torch.mean(user_combined, dim=1)
            
            # Combine embeddings for items
            item_combined = torch.stack([
                item_item_embeddings,
                user_item_embeddings_item
            ], dim=1)
            item_embeddings = torch.mean(item_combined, dim=1)
            
            # Combine all
            cur_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
            all_embeddings.append(cur_embeddings)

        # Mean of all layer embeddings
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[:self.user_num]
        item_all_embeddings = all_embeddings[self.user_num:]

        return user_all_embeddings, item_all_embeddings