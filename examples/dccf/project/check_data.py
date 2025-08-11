import pickle
import os

if __name__ == "__main__":
    data_path = '/workplace/project/data/gowalla'
    print("Current directory:", os.getcwd())
    print("Data files:", os.listdir(data_path))
    
    print("\nLoading training matrix...")
    with open(os.path.join(data_path, 'trnMat.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    print("Training matrix shape:", train_data.shape)
    print("Training matrix type:", type(train_data))
    
    print("\nLoading test matrix...")
    with open(os.path.join(data_path, 'tstMat.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    print("Test matrix shape:", test_data.shape)
    print("Test matrix type:", type(test_data))