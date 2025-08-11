import numpy as np
import torch
import scipy.sparse as sp
import pickle
import os

def load_data(data_dir):
    """Load Yelp dataset"""
    with open(os.path.join(data_dir, 'trnMat.pkl'), 'rb') as f:
        train_mat = pickle.load(f)
    with open(os.path.join(data_dir, 'tstMat.pkl'), 'rb') as f:
        test_mat = pickle.load(f)
    
    return train_mat, test_mat

def create_adj_matrices(train_mat):
    """Create adjacency matrices for heterogeneous graphs"""
    # User-Item interaction graph
    user_item_graph = train_mat.tocoo()
    values = user_item_graph.data
    indices = np.vstack((user_item_graph.row, user_item_graph.col))
    user_item_graph = torch.sparse_coo_tensor(indices, values, user_item_graph.shape)

    # User-User graph through common items
    user_user = train_mat @ train_mat.T
    user_user.setdiag(0)
    user_user.eliminate_zeros()
    # Normalize
    user_user = normalize_adj_matrix(user_user)
    user_user = sp_mat_to_torch_sparse(user_user)

    # Item-Item graph through common users
    item_item = train_mat.T @ train_mat
    item_item.setdiag(0)
    item_item.eliminate_zeros()
    # Normalize
    item_item = normalize_adj_matrix(item_item)
    item_item = sp_mat_to_torch_sparse(item_item)

    return user_item_graph, user_user, item_item

def normalize_adj_matrix(adj):
    """Symmetrically normalize adjacency matrix"""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

def sp_mat_to_torch_sparse(sp_mat):
    """Convert scipy sparse matrix to torch sparse tensor"""
    coo = sp_mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    return torch.sparse_coo_tensor(indices, values, coo.shape)

class RecommenderDataset:
    def __init__(self, train_mat, batch_size=2048, num_negatives=1):
        self.train_mat = train_mat
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        
        self.num_users, self.num_items = train_mat.shape
        self.train_items = [[] for _ in range(self.num_users)]
        
        # Convert to list format and record positive items for each user
        for (u, i) in zip(*train_mat.nonzero()):
            self.train_items[u].append(i)

    def sample(self):
        """Sample a batch of training instances"""
        users, pos_items, neg_items = [], [], []
        
        for _ in range(self.batch_size):
            user = np.random.randint(self.num_users)
            pos_item = np.random.choice(self.train_items[user])
            
            # Negative sampling
            while True:
                neg_item = np.random.randint(self.num_items)
                if neg_item not in self.train_items[user]:
                    break
                    
            users.append(user)
            pos_items.append(pos_item)
            neg_items.append(neg_item)
            
        return torch.LongTensor(users), torch.LongTensor(pos_items), torch.LongTensor(neg_items)