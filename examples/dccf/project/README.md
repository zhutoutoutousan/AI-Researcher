# Intent-Aware Graph Contrastive Learning for Recommendation

This project implements a self-supervised learning approach for recommendation systems that combines Graph Neural Networks (GNNs), disentangled intent modeling, and contrastive learning.

## Project Structure

```
project/
├── data/
│   └── gowalla/            # Gowalla dataset
├── data_processing/
│   └── dataset.py          # Dataset loading and processing
├── model/
│   ├── intentgcl.py        # Main model implementation
│   └── utils.py            # Utility functions
└── run_training_testing.py # Main training script
```

## Core Components

1. **Intent-Aware GNN**
   - Multiple GNN layers for message passing
   - Learnable intent prototypes
   - Attention-based intent aggregation

2. **Disentangled Representations**
   - Intent prototype learning
   - Multi-intent attention mechanism
   - SVD-enhanced message passing

3. **Contrastive Learning**
   - InfoNCE loss with temperature scaling
   - Local and global view contrasting
   - Intent-aware negative sampling

## Model Architecture

The model combines three key components:

1. **Graph Neural Network Layers**
```python
# Message passing aggregation
msg_user = torch.spmm(adj_norm, item_embeddings)
msg_item = torch.spmm(adj_norm.t(), user_embeddings)
```

2. **Intent Modeling**
```python
# Intent attention computation
attention = softmax(embeddings @ intent_prototypes.t() / temp)
intent_aware_emb = attention @ intent_prototypes
```

3. **Contrastive Learning**
```python
# Contrastive loss with temperature scaling
pos_score = (intent_emb * user_emb).sum(1) / temp
neg_score = log(exp(intent_emb @ all_emb.t() / temp).sum(1))
```

## Usage

1. **Data Preparation**
   - The Gowalla dataset is used
   - Processed into training and testing matrices
   - Normalized adjacency matrix computation

2. **Training**
```bash
python run_training_testing.py
```

3. **Configuration**
   - Embedding dimension: 64
   - Number of GNN layers: 2
   - Number of intents: 128
   - Temperature: 0.2
   - Learning rate: 0.001

## Results

The model is evaluated using standard recommendation metrics:
- Recall@20, NDCG@20
- Recall@40, NDCG@40

Results are saved after training with timestamp.

## References

The implementation is based on:
1. LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation
2. Learning intents behind interactions with knowledge graph for recommendation
3. Neural Graph Collaborative Filtering