import argparse
import torch
import os
import sys
from utils.train import train_gnn, test_gnn
from utils.weights import get_class_weights_tensor
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv,SAGEConv, GCNConv
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import random
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser(description='GNN Benchmark Script')
parser.add_argument('--model', type=str, choices=['SAGE', 'GAT', 'GCN'], help='Choose GNN model: SAGE, GAT, or GCN')
parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
parser.add_argument('--hidden_features', type=int, default=64, help='Number of hidden features')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
parser.add_argument('--in_heads', type=int, default=8, help='Number of in heads (for GAT)')
parser.add_argument('--out_heads', type=int, default=8, help='Number of out heads (for GAT)')
parser.add_argument('--input_dim', type=int, help='Input dimension for GNN models')
parser.add_argument('--output_dim', type=int, help='Output dimension for GNN models')
parser.add_argument('--max_num_epochs', type=int, default=100, help='Maximum number of epochs')
parser.add_argument('--start_lr', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--num_repetitions', type=int, default=5, help='Number of repetitions')
args = parser.parse_args()
print(sys.argv)

# Set seeds for reproducibility
torch.cuda.empty_cache()
seed_value = 77
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"
INPUT_GRAPH = 'data/A_graph.pt' #data/sub_graph.pt  
INPUT_FOLDER = 'data/data_original/' #data/data_original_sub/ to use sub-population dataset

# Check if a GPU is available
if torch.cuda.is_available():
    # Get the current GPU device
    device = torch.cuda.current_device()
    
    # Get the GPU's memory usage in bytes
    memory_allocated = torch.cuda.memory_allocated(device)
    memory_cached = torch.cuda.memory_cached(device)
    
    # Convert bytes to a more human-readable format (e.g., megabytes or gigabytes)
    memory_allocated_mb = memory_allocated / 1024**2  # Megabytes
    memory_cached_mb = memory_cached / 1024**2  # Megabytes
    
    print(f"GPU Memory Allocated: {memory_allocated_mb:.2f} MB")
    print(f"GPU Memory Cached: {memory_cached_mb:.2f} MB")
else:
    print("No GPU available.")

label0_count=[]
for j in range(30):  
    df = pd.read_csv(f"{INPUT_FOLDER}Case_{j+1}.csv") 
    label0_count.append(len(df))

class_weights_tensor = get_class_weights_tensor(INPUT_GRAPH)
class_weights_tensor = [weights.to(device) for weights in class_weights_tensor]

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNNModel, self).__init__()
        self.num_layers = args.num_layers
        self.hidden_features = args.hidden_features
        self.dropout = args.dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        if args.model == 'SAGE':
            self.conv = self.make_sage_layers(input_dim, output_dim)
        elif args.model == 'GAT':
            self.conv = self.make_gat_layers(input_dim, output_dim)
        elif args.model == 'GCN':
            self.conv = self.make_gcn_layers(input_dim, output_dim)
    
    def make_sage_layers(self, input_dim, output_dim):
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(SAGEConv(input_dim, self.hidden_features))
            else:
                layers.append(SAGEConv(self.hidden_features, self.hidden_features))
        layers.append(SAGEConv(self.hidden_features, output_dim))
        return torch.nn.ModuleList(layers)
    
    def make_gat_layers(self, input_dim, output_dim):
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(GATConv(input_dim, self.hidden_features, heads=args.in_heads, dropout=self.dropout))
            else:
                layers.append(GATConv(self.hidden_features * args.in_heads, self.hidden_features, heads=args.out_heads, dropout=self.dropout))
        layers.append(GATConv(self.hidden_features * args.in_heads, output_dim))
        return torch.nn.ModuleList(layers)
    
    def make_gcn_layers(self, input_dim, output_dim):
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(GCNConv(input_dim, self.hidden_features))
            else:
                layers.append(GCNConv(self.hidden_features, self.hidden_features))
        layers.append(GCNConv(self.hidden_features, output_dim))
        return torch.nn.ModuleList(layers)
    
    def reset_parameters(self):
        for conv in self.conv:
            conv.reset_parameters()
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.conv:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

class MyGraphDataset(Dataset):
    def __init__(self, num_samples, transform=None, pre_transform=None):
        super(MyGraphDataset, self).__init__(transform, pre_transform)
        self.num_samples = num_samples
        self.data_list = torch.load(INPUT_GRAPH)
        self.class_weights = class_weights_tensor  
    
    def len(self):
        return self.num_samples

    def get(self, idx):
        return self.data_list[idx]

def gnn_evaluation(gnn, max_num_epochs, batch_size, start_lr, num_repetitions, min_lr=0.000001, factor=0.04, patience=7):
    dataset = MyGraphDataset(num_samples=len(torch.load(INPUT_GRAPH))).shuffle()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_model_state_dict = None
    patient_dict=dict()
    for i in range(num_repetitions):
        kf = KFold(n_splits=7, shuffle=True)
        dataset.shuffle()

        for train_index, test_index in kf.split(list(range(len(dataset)))):
            train_index, val_index = train_test_split(train_index, test_size=0.1)

            train_dataset = dataset[train_index.tolist()]
            val_dataset = dataset[val_index.tolist()]
            test_dataset = dataset[test_index.tolist()]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            num_patients = len(test_loader)
            print(f"Number of patients in the test loader: {num_patients}")

            input_dim = dataset[0].x.shape[1]
            output_dim = len(torch.unique(dataset[0].y))
            model = gnn(input_dim, output_dim).to(device)
            model.reset_parameters()

            optimizer = torch.optim.Adam(model.parameters(), lr=start_lr, weight_decay=0.0005)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor,
                                                                   patience=patience, min_lr=0.0000001)

            best_val_acc = 0.0
            best_model_state_dict = None
            early_stopping_counter = 0
            early_stopping_patience = 50

            for epoch in range(1, max_num_epochs + 1):
                lr = scheduler.optimizer.param_groups[0]['lr']
                torch.cuda.empty_cache()
                train_gnn(train_loader, model, optimizer, device, class_weights_tensor)
                val_acc = test_gnn(val_loader, model, device)
                scheduler.step(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_acc = test_gnn(test_loader, model, device) * 100.0
                    best_model_state_dict = model.state_dict()
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        print("Early stopping triggered. No improvement in validation accuracy for {} epochs.".format(early_stopping_patience))
                        break

            model.load_state_dict(best_model_state_dict)
            #torch.save(model.state_dict(), 'gat64.pt')
            # Evaluate on the entire test set
            model.eval()
            for data in test_loader:
                data = data.to(device)
                output = model(data)
                predss=output.max(dim=1)[1].cpu().numpy()
                labelss=data.y.cpu().numpy()
                idx=label0_count.index(len(labelss))+1
                precision, recall, f1, _ = precision_recall_fscore_support(labelss, predss, average='weighted', zero_division=1)
                if idx not in patient_dict.keys():
                    patient_dict[idx]=dict()
                    patient_dict[idx]['f1']=[f1]
                    patient_dict[idx]['pred']=[predss]
                    patient_dict[idx]['label']=[labelss]
                else:
                    patient_dict[idx]['f1'].append(f1)
                    patient_dict[idx]['pred'].append(predss)
                    patient_dict[idx]['label'].append(labelss)

    return patient_dict

patient_dict = gnn_evaluation(GNNModel, args.max_num_epochs, batch_size=1, start_lr=args.start_lr, num_repetitions=args.num_repetitions)

# Initialize a list to store the ratios for each label across all patients
average_ratio_per_label = []
percentage_corrected_labels = []
results_dir = "res_ind_gnn"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
results_file = os.path.join(results_dir, "res_ind_gnn.txt")
with open(results_file, "w") as f:
    for key in patient_dict.keys():
        idx = patient_dict[key]['f1'].index(max(patient_dict[key]['f1']))

        # Compute metrics for each patient
        conf_matrix = confusion_matrix(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx])
        precision = precision_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1)
        recall = recall_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1)
        accuracy = accuracy_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx])
        f1 = f1_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1)

        f.write(f"Metrics for Patient {key}:\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

        average_precision = np.mean([precision_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1) for key in patient_dict.keys()])
        average_recall = np.mean([recall_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1) for key in patient_dict.keys()])
        average_f1 = np.mean([f1_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1) for key in patient_dict.keys()])

        f.write("\nAverage Metrics Across All Patients:\n")
        f.write(f"Average Precision: {average_precision:.4f}\n")
        f.write(f"Average Recall: {average_recall:.4f}\n")
        f.write(f"Average F1 Score: {average_f1:.4f}\n")

        total_right_cells = np.sum(np.diag(conf_matrix))
        ratio_per_label = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)

        for i, label in enumerate(range(conf_matrix.shape[0])):
            f.write(f"Label {label}:\n")
            f.write(f"Ratio of Correct Predictions: {ratio_per_label[i]:.4f}\n")

            if len(average_ratio_per_label) <= i:
                average_ratio_per_label.append([ratio_per_label[i]])
            else:
                average_ratio_per_label[i].append(ratio_per_label[i])

        f.write("-" * 50 + "\n")

        percentage_corrected = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
        percentage_corrected_labels.append(percentage_corrected)
average_ratio_per_label = np.mean(average_ratio_per_label, axis=1)

f.write("\nAverage Ratios Across All Patients:\n")
label_dict = {0: 'O', 1: 'N', 2: 'G', 3: 'P', 4: 'K', 5: 'B'}
for i, average_ratio in enumerate(average_ratio_per_label):
    f.write(f"Label {label_dict[i]}: {average_ratio:.4f}\n")
