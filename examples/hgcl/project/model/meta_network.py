import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rank=10):
        super(MetaNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rank = rank

        # Low-rank decomposition for transformation matrices
        self.fc1_u = nn.Linear(input_dim, rank)
        self.fc1_v = nn.Linear(rank, hidden_dim * input_dim)
        self.prelu1 = nn.PReLU()
        
        self.fc2_u = nn.Linear(hidden_dim, rank)
        self.fc2_v = nn.Linear(rank, output_dim * hidden_dim)
        self.prelu2 = nn.PReLU()

    def get_transformation_matrix(self, embedding):
        # First layer transformation
        h1_u = self.fc1_u(embedding)
        h1_v = self.fc1_v(h1_u)
        h1 = h1_v.view(-1, self.input_dim, self.hidden_dim)
        h1 = self.prelu1(h1)

        # Second layer transformation
        h2_u = self.fc2_u(h1.mean(dim=1))
        h2_v = self.fc2_v(h2_u)
        h2 = h2_v.view(-1, self.hidden_dim, self.output_dim)
        h2 = self.prelu2(h2)

        return torch.bmm(h1, h2)  # Final transformation matrix

    def forward(self, user_embeddings, item_embeddings):
        # Generate personalized transformation matrices
        user_trans = self.get_transformation_matrix(user_embeddings)
        item_trans = self.get_transformation_matrix(item_embeddings)

        # Apply transformations
        user_transformed = torch.bmm(user_embeddings.unsqueeze(1), user_trans).squeeze(1)
        item_transformed = torch.bmm(item_embeddings.unsqueeze(1), item_trans).squeeze(1)

        return user_transformed, item_transformed