import torch
import torch.nn as nn
import torch.nn.functional as F
from .heterogeneous_gnn import HeterogeneousGNN
from .meta_network import MetaNetwork

class HeteroContrastiveModel(nn.Module):
    def __init__(self, user_num, item_num, embed_dim, n_layers=2, temp=0.1, lambda_coef=0.5):
        super(HeteroContrastiveModel, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.embed_dim = embed_dim
        self.temp = temp
        self.lambda_coef = lambda_coef

        # Heterogeneous GNN for main view
        self.main_gnn = HeterogeneousGNN(user_num, item_num, embed_dim, n_layers)
        
        # Meta network for personalized knowledge transfer
        self.meta_net = MetaNetwork(embed_dim, embed_dim*2, embed_dim)

    def forward(self, user_graph, user_item_graph, item_graph):
        # Get main view embeddings
        user_emb_main, item_emb_main = self.main_gnn(user_graph, user_item_graph, item_graph)
        
        # Generate auxiliary view through meta network
        user_emb_aux, item_emb_aux = self.meta_net(user_emb_main, item_emb_main)

        return user_emb_main, item_emb_main, user_emb_aux, item_emb_aux

    def calc_ssl_loss(self, user_emb1, user_emb2, temp):
        """Calculate contrastive loss for self-supervised learning"""
        # Normalize embeddings
        user_emb1 = F.normalize(user_emb1, dim=1)
        user_emb2 = F.normalize(user_emb2, dim=1)
        
        # Similarity matrix
        pos_score = (user_emb1 * user_emb2).sum(dim=1)
        pos_score = torch.exp(pos_score / temp)
        ttl_score = torch.matmul(user_emb1, user_emb2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temp).sum(dim=1)

        ssl_loss = -torch.log(pos_score / ttl_score + 1e-8)
        return ssl_loss.mean()

    def calc_bpr_loss(self, user_emb, item_emb, users, pos_items, neg_items):
        """Calculate Bayesian Personalized Ranking loss"""
        users_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
        return bpr_loss.mean()

    def calculate_loss(self, user_graph, user_item_graph, item_graph, 
                      users, pos_items, neg_items):
        """Calculate total loss combining BPR and contrastive loss"""
        # Get embeddings from both views
        user_emb_main, item_emb_main, user_emb_aux, item_emb_aux = self.forward(
            user_graph, user_item_graph, item_graph)

        # BPR loss on main embeddings
        bpr_loss = self.calc_bpr_loss(user_emb_main, item_emb_main, 
                                     users, pos_items, neg_items)

        # Contrastive loss between main and auxiliary views
        ssl_loss = self.calc_ssl_loss(user_emb_main, user_emb_aux, self.temp)

        # Total loss
        loss = bpr_loss + self.lambda_coef * ssl_loss
        return loss