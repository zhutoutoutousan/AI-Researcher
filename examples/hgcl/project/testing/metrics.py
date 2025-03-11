import numpy as np

def recall_and_ndcg_at_k(predictions, test_labels, k):
    """Calculate Recall@K and NDCG@K"""
    user_num = 0
    recall = 0
    ndcg = 0
    
    for user in range(len(predictions)):
        if user in test_labels:
            items = test_labels[user]
            
            # Get top k predictions
            pred = predictions[user]
            pred[pred < -1e8] = -1e8  # Handle overflow
            pos = set(items)
            items_idx = np.zeros_like(pred, dtype=bool)
            items_idx[items] = True
            
            # Calculate recall
            topk_idx = np.argpartition(pred, -k)[-k:]
            topk_idx = topk_idx[np.argsort(-pred[topk_idx])]
            hit_num = len(set(topk_idx) & pos)
            recall += hit_num / len(pos)
            
            # Calculate NDCG
            dcg = 0
            idcg = np.sum(1 / np.log2(np.arange(2, min(k, len(pos)) + 2)))
            if idcg == 0:
                continue
            
            for i, item_idx in enumerate(topk_idx):
                if item_idx in pos:
                    dcg += 1 / np.log2(i + 2)
            ndcg += dcg / idcg
            
            user_num += 1
    
    return recall / user_num, ndcg / user_num