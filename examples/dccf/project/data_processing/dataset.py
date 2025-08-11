import numpy as np
import torch
import pickle
import torch.utils.data as data

class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(self.dokmat.shape[1])
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

def load_data(path):
    # Load training data
    with open(path + 'trnMat.pkl', 'rb') as f:
        train = pickle.load(f)
    train_csr = (train != 0).astype(np.float32)
    
    # Load test data
    with open(path + 'tstMat.pkl', 'rb') as f:
        test = pickle.load(f)

    # Normalize adjacency matrix
    rowD = np.array(train.sum(1)).squeeze()
    colD = np.array(train.sum(0)).squeeze()
    for i in range(len(train.data)):
        train.data[i] = train.data[i] / pow(rowD[train.row[i]] * colD[train.col[i]], 0.5)

    # Process test set
    test_labels = [[] for _ in range(test.shape[0])]
    for i in range(len(test.data)):
        row = test.row[i]
        col = test.col[i]
        test_labels[row].append(col)

    return train.tocoo(), train_csr, test_labels