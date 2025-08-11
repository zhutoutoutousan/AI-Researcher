import os
import numpy as np
import pickle
import torch
from collections import defaultdict

def convert_to_matrix(interaction_dict, num_users, num_items):
    """Convert interaction dictionary to sparse matrix"""
    matrix = torch.zeros((num_users, num_items))
    for user_id, item_sets in interaction_dict.items():
        for item_id in item_sets:
            if user_id < num_users and item_id < num_items:
                matrix[user_id][item_id] = 1.0
    return matrix

def process_interaction_data(data_arr, max_users=10000, max_items=10000):
    """Process interaction data and convert to matrix format"""
    interaction_dict = data_arr[0]  # Get the first dictionary
    
    # Get maximum user and item IDs
    users = set()
    items = set()
    for user_id, item_sets in interaction_dict.items():
        if user_id < max_users:
            users.add(user_id)
            for item_id in item_sets:
                if item_id < max_items:
                    items.add(item_id)
    
    num_users = min(max(users) + 1, max_users)
    num_items = min(max(items) + 1, max_items)
    
    print(f"Number of users: {num_users}, Number of items: {num_items}")
    
    # Convert to matrix
    return convert_to_matrix(interaction_dict, num_users, num_items)

def load_and_convert_data(source_dir, target_dir, max_users=10000, max_items=10000):
    """Load data from numpy files and convert to pickle format with size limits"""
    os.makedirs(target_dir, exist_ok=True)
    print(f"Loading data from {source_dir}")
    
    # Load training and testing data
    train_data = np.load(os.path.join(source_dir, "training_set.npy"), allow_pickle=True)
    test_data = np.load(os.path.join(source_dir, "testing_set.npy"), allow_pickle=True)
    
    print("\nProcessing training data...")
    train_matrix = process_interaction_data(train_data, max_users, max_items)
    print("\nProcessing testing data...")
    test_matrix = process_interaction_data(test_data, max_users, max_items)
    
    print(f"\nFinal shapes:")
    print(f"Train matrix: {train_matrix.shape}")
    print(f"Test matrix: {test_matrix.shape}")
    
    # Save tensors
    print(f"\nSaving to {target_dir}")
    with open(os.path.join(target_dir, "trnMat.pkl"), 'wb') as f:
        pickle.dump(train_matrix, f)
    with open(os.path.join(target_dir, "tstMat.pkl"), 'wb') as f:
        pickle.dump(test_matrix, f)
    
    print("Data preparation completed!")

if __name__ == "__main__":
    source_dir = "/workplace/reference_repos/LR-GCCF/data/gowalla/datanpy"
    target_dir = "/workplace/project/data/gowalla"
    max_users = 5000  # Limit size to avoid memory issues
    max_items = 5000
    
    load_and_convert_data(source_dir, target_dir, max_users, max_items)