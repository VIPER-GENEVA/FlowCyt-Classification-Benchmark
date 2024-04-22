import argparse
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
import os

def process_data(in_dir, out_dir):
    data_FC = []
    for j in range(30):
        df = pd.read_csv(os.path.join(in_dir, f"Case_{j+1}.csv"))
        labels = df[["label"]]
        df = df.drop('label', axis=1)
        x = df[[ 'FS INT',  'SS INT', 'FL1 INT_CD14-FITC', 'FL2 INT_CD19-PE', 'FL3 INT_CD13-ECD', 'FL4 INT_CD33-PC5.5', 'FL5 INT_CD34-PC7', 'FL6 INT_CD117-APC', 'FL7 INT_CD7-APC700', 'FL8 INT_CD16-APC750', 'FL9 INT_HLA-PB', 'FL10 INT_CD45-KO']].to_numpy()

        k_neighbors = 7  # Number of neighbors for k-NN
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean').fit(x)
        distances, indices = nbrs.kneighbors(x)
        src = np.repeat(np.arange(x.shape[0]), k_neighbors)
        dst = indices.flatten()
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # Min-Max Normalization
        for i in range(x.shape[1]):
            x[:,i] = (x[:,i] - x[:,i].min()) / (x[:,i].max() - x[:,i].min())
        data_FC.append(Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index, y=torch.tensor(list(labels.to_numpy().flatten()))))

    torch.save(data_FC, out_dir)

def main():
    parser = argparse.ArgumentParser(description='Generate graph data')
    parser.add_argument('--in_dir', type=str, required=True,
                        help='Directory containing the data files')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output file for the generated graph data')
    args = parser.parse_args()
    if not os.path.exists(args.in_dir):
        print(f"Error: Data directory '{args.in_dir}' does not exist.")
        return
    process_data(args.in_dir, args.out_dir)

if __name__ == "__main__":
    main()