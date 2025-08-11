import torch
import torch.nn as nn
import torch.nn.functional as F

class IntentModule(nn.Module):
    def __init__(self, embed_dim, n_intents, temp=0.2, use_multi_head=True, n_heads=4):
        super(IntentModule, self).__init__()
        self.embed_dim = embed_dim
        self.n_intents = n_intents
        self.temp = temp
        self.use_multi_head = use_multi_head
        self.n_heads = n_heads
        
        # Initialize intent prototypes
        self.intent_prototypes = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(n_intents, embed_dim))
        )
        
        if use_multi_head:
            self.head_dim = embed_dim // n_heads
            self.attention_heads = nn.ModuleList([
                nn.Linear(embed_dim, self.head_dim) for _ in range(n_heads)
            ])
            self.head_projections = nn.ModuleList([
                nn.Linear(embed_dim, self.head_dim) for _ in range(n_heads)
            ])
            self.head_combine = nn.Linear(n_intents * n_heads, n_intents, bias=False)
            
        # Learnable temperature
        self.temp_param = nn.Parameter(torch.ones(1) * temp)
        
    def compute_attention(self, embeddings):
        if self.use_multi_head:
            # Multi-head attention
            head_outputs = []
            for head, proj in zip(self.attention_heads, self.head_projections):
                # Project embeddings and prototypes to head dimension
                head_q = head(embeddings)  # [batch_size, head_dim]
                head_k = proj(self.intent_prototypes)  # [n_intents, head_dim]
                
                # Compute attention scores
                attention = torch.matmul(head_q, head_k.t())  # [batch_size, n_intents]
                attention = F.softmax(attention / self.temp_param, dim=1)
                head_outputs.append(attention)
            
            # Concatenate heads
            attention = torch.cat(head_outputs, dim=1)  # [batch_size, n_intents * n_heads]
            
            # Project back to n_intents dimension
            attention = self.head_combine(attention)  # [batch_size, n_intents]
            attention = F.softmax(attention / self.temp_param, dim=1)
        else:
            # Simple attention
            attention = torch.matmul(embeddings, self.intent_prototypes.t())
            attention = F.softmax(attention / self.temp_param, dim=1)
        
        return attention
        
    def forward(self, embeddings):
        # Compute attention scores
        attention = self.compute_attention(embeddings)
        
        # Generate intent-aware representations
        intent_aware_emb = torch.matmul(attention, self.intent_prototypes)
        
        # Residual connection and layer normalization
        output = embeddings + intent_aware_emb
        output = F.layer_norm(output, output.size()[1:])
        
        return output, attention
        
    def get_regularization(self):
        # L2 regularization for intent prototypes and attention modules
        reg_loss = torch.sum(self.intent_prototypes.pow(2))
        if self.use_multi_head:
            for head, proj in zip(self.attention_heads, self.head_projections):
                reg_loss = reg_loss + torch.sum(head.weight.pow(2))
                reg_loss = reg_loss + torch.sum(proj.weight.pow(2))
            reg_loss = reg_loss + torch.sum(self.head_combine.weight.pow(2))
        return reg_loss