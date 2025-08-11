import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import sparse_dropout, spmm
from model.intent_module import IntentModule

class IntentGCL(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm,
                 n_layers, temp, lambda_1, lambda_2, lambda_3, dropout, n_intents, batch_user, device,
                 use_residual=True):
        super(IntentGCL, self).__init__()
        
        # Parameters and configurations
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.device = device
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.use_residual = use_residual
        
        # SVD components
        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt
        
        # Training related
        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.dropout = dropout
        
        # Initialize embeddings
        self.user_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_users, embed_dim)))
        self.item_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_items, embed_dim)))
        
        # Intent module
        self.intent_module = IntentModule(embed_dim, n_intents, temp=temp)
        
        # Layer weight matrices
        self.W = nn.ModuleList()
        for _ in range(n_layers):
            layer_weight = nn.Linear(embed_dim, embed_dim, bias=True)
            nn.init.xavier_uniform_(layer_weight.weight)
            self.W.append(layer_weight)
        
        # Layer storage
        self.user_embeddings = [None] * (n_layers + 1)
        self.item_embeddings = [None] * (n_layers + 1)
        self.msg_user_embeddings = [None] * (n_layers + 1)
        self.msg_item_embeddings = [None] * (n_layers + 1)
        self.intent_user_embeddings = [None] * (n_layers + 1)
        self.intent_item_embeddings = [None] * (n_layers + 1)
        
        # Output embeddings
        self.final_user_embedding = None
        self.final_item_embedding = None

    def _message_passing(self, layer_idx):
        if layer_idx == 0:
            self.user_embeddings[0] = self.user_embedding
            self.item_embeddings[0] = self.item_embedding
            return

        # Message passing
        self.msg_user_embeddings[layer_idx] = spmm(
            sparse_dropout(self.adj_norm, self.dropout),
            self.item_embeddings[layer_idx-1],
            self.device
        )
        self.msg_item_embeddings[layer_idx] = spmm(
            sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1),
            self.user_embeddings[layer_idx-1],
            self.device
        )

        # SVD enhancement
        vt_ei = torch.matmul(self.vt, self.item_embeddings[layer_idx-1])
        self.intent_user_embeddings[layer_idx] = torch.matmul(self.u_mul_s, vt_ei)
        ut_eu = torch.matmul(self.ut, self.user_embeddings[layer_idx-1])
        self.intent_item_embeddings[layer_idx] = torch.matmul(self.v_mul_s, ut_eu)

        # Intent-aware transformation
        msg_user_emb, _ = self.intent_module(self.msg_user_embeddings[layer_idx])
        msg_item_emb, _ = self.intent_module(self.msg_item_embeddings[layer_idx])

        # Layer transformation
        transformed_user = self.W[layer_idx-1](msg_user_emb)
        transformed_item = self.W[layer_idx-1](msg_item_emb)

        # Residual connection
        if self.use_residual:
            self.user_embeddings[layer_idx] = transformed_user + self.user_embeddings[layer_idx-1]
            self.item_embeddings[layer_idx] = transformed_item + self.item_embeddings[layer_idx-1]
        else:
            self.user_embeddings[layer_idx] = transformed_user
            self.item_embeddings[layer_idx] = transformed_item

        # Normalization
        self.user_embeddings[layer_idx] = F.layer_norm(self.user_embeddings[layer_idx], 
                                                      self.user_embeddings[layer_idx].size()[1:])
        self.item_embeddings[layer_idx] = F.layer_norm(self.item_embeddings[layer_idx], 
                                                      self.item_embeddings[layer_idx].size()[1:])

    def _aggregate_embeddings(self):
        # Aggregate embeddings from all layers with learnable weights
        user_embs = torch.stack(self.user_embeddings)
        item_embs = torch.stack(self.item_embeddings)
        
        # Final embeddings
        self.final_user_embedding = torch.mean(user_embs, dim=0)
        self.final_item_embedding = torch.mean(item_embs, dim=0)

    def forward(self, user_ids, item_ids, pos_items, neg_items, test=False):
        # Message passing for all layers
        for layer in range(self.n_layers + 1):
            self._message_passing(layer)
        
        # Aggregate embeddings
        self._aggregate_embeddings()
        
        if test:
            return self.final_user_embedding[user_ids] @ self.final_item_embedding.t()
        
        # Get user and item embeddings for loss computation
        user_emb = self.final_user_embedding[user_ids]
        pos_emb = self.final_item_embedding[pos_items]
        neg_emb = self.final_item_embedding[neg_items]
        
        # Compute BPR loss
        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)
        loss_bpr = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        
        # Compute contrastive loss
        user_augmented = torch.stack(self.msg_user_embeddings[1:]).mean(0)[user_ids]
        item_augmented = torch.stack(self.msg_item_embeddings[1:]).mean(0)[item_ids]
        
        sim_matrix = torch.matmul(user_augmented, item_augmented.t()) / self.temp
        labels = torch.arange(len(user_ids)).to(self.device)
        loss_contrast = F.cross_entropy(sim_matrix, labels)
        
        # L2 regularization
        reg_loss = self.lambda_3 * (
            torch.norm(user_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb) +
            self.intent_module.get_regularization()
        )
        
        # Total loss
        loss = self.lambda_1 * loss_bpr + self.lambda_2 * loss_contrast + reg_loss
        
        return loss, loss_bpr, loss_contrast, reg_loss