import argparse
import torch
import os
import sys
from utils.train import train_dnn, test_dnn
from utils.weights import get_class_weights_tensor
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import random
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set seeds for reproducibility
torch.cuda.empty_cache()
seed_value = 77
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

PRINT_MEMORY = False
device_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)
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

class MyGraphDataset(Dataset):
    def __init__(self,  num_samples,transform=None, pre_transform=None):
        super(MyGraphDataset, self).__init__(transform, pre_transform)
        self.num_samples = num_samples
        self.data_list = torch.load(INPUT_GRAPH)
        self.class_weights = class_weights_tensor  
        
    def len(self):
        return self.num_samples

    def get(self, idx):
        return self.data_list[idx]

class NeuralNetwork(nn.Module):
    def __init__(self, num_layers=3, hidden_dim=128, input_dim=12, output_dim=6, dropout=0.4):
        super(NeuralNetwork, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        for _ in range(self.num_layers - 2):
            self.fc_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc_layers.append(nn.Linear(self.hidden_dim, self.output_dim))

    def reset_parameters(self):
        for layer in self.fc_layers:
            layer.reset_parameters()

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.fc_layers[i](x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc_layers[-1](x)
        return F.log_softmax(x, dim=1)

    def __repr__(self):
        return self.__class__.__name__

def nn_evaluation(num_layers=3, hidden_dim=128, dropout=0.4, max_num_epochs=200, batch_size=128, start_lr=0.01,
                   min_lr=0.000001, factor=0.5, patience=5, num_repetitions=10, all_std=True, input_dim=12, output_dim=6):
    dataset = MyGraphDataset(num_samples=len(torch.load(INPUT_GRAPH))).shuffle()  # data/sub_graph.pt   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_model_state_dict = None
    patient_dict = dict()
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

            model = NeuralNetwork(num_layers=num_layers, hidden_dim=hidden_dim, input_dim=input_dim, output_dim=output_dim, dropout=dropout).to(device)
            model.reset_parameters()

            optimizer = torch.optim.Adam(model.parameters(), lr=start_lr, weight_decay=0.005)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor,
                                                                   patience=patience, min_lr=0.0000001)

            best_val_acc = 0.0

            early_stopping_counter = 0
            early_stopping_threshold = 50  # Number of epochs without improvement to trigger early stopping

            for epoch in range(1, max_num_epochs + 1):
                lr = scheduler.optimizer.param_groups[0]['lr']
                torch.cuda.empty_cache()
                train_dnn(train_loader, model, optimizer, device, class_weights_tensor)
                val_acc = test_dnn(val_loader, model, device)
                scheduler.step(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_acc = test_dnn(test_loader, model, device) * 100.0
                    best_model_state_dict = model.state_dict()
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= early_stopping_threshold:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            model.load_state_dict(best_model_state_dict)
            # Evaluate on the entire test set
            model.eval()
            for data in test_loader:
                matrix = data.x.to(device)
                output = model(matrix)
                predss = output.max(dim=1)[1].cpu().numpy()
                labelss = data.y.cpu().numpy()
                idx = label0_count.index(len(labelss)) + 1
                precision, recall, f1, _ = precision_recall_fscore_support(labelss, predss, average='weighted',
                                                                           zero_division=1)

                if idx not in patient_dict.keys():
                    patient_dict[idx] = dict()
                    patient_dict[idx]['f1'] = [f1]
                    patient_dict[idx]['pred'] = [predss]
                    patient_dict[idx]['label'] = [labelss]
                else:
                    patient_dict[idx]['f1'].append(f1)
                    patient_dict[idx]['pred'].append(predss)
                    patient_dict[idx]['label'].append(labelss)

    return patient_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Network Benchmark Script')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout probability')
    parser.add_argument('--max_num_epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--start_lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--num_repetitions', type=int, default=7, help='Number of repetitions')
    parser.add_argument('--input_dim', type=int, default=12, help='Number of input features')
    parser.add_argument('--output_dim', type=int, default=6, help='Number of output features')
    args = parser.parse_args()
    print(sys.argv)
    patient_dict = nn_evaluation(num_layers=args.num_layers, hidden_dim=args.hidden_dim, dropout=args.dropout,
                                 max_num_epochs=args.max_num_epochs, batch_size=args.batch_size,
                                 start_lr=args.start_lr, num_repetitions=args.num_repetitions,
                                 input_dim=args.input_dim, output_dim=args.output_dim)

# Initialize a list to store the ratios for each label across all patients
average_ratio_per_label = []
percentage_corrected_labels = []
for key in patient_dict.keys():
        idx = patient_dict[key]['f1'].index(max(patient_dict[key]['f1']))

        # Compute metrics for each patient
        conf_matrix = confusion_matrix(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx])
        precision = precision_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1)
        recall = recall_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1)
        accuracy = accuracy_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx])
        f1 = f1_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1)

        print(f"Metrics for Patient {key}:")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        average_precision = np.mean([precision_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1) for key in patient_dict.keys()])
        average_recall = np.mean([recall_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1) for key in patient_dict.keys()])
        average_f1 = np.mean([f1_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1) for key in patient_dict.keys()])

        print("\nAverage Metrics Across All Patients:")
        print(f"Average Precision: {average_precision:.4f}")
        print(f"Average Recall: {average_recall:.4f}")
        print(f"Average F1 Score: {average_f1:.4f}")

        total_right_cells = np.sum(np.diag(conf_matrix))
        ratio_per_label = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)

        for i, label in enumerate(range(conf_matrix.shape[0])):
            print(f"Label {label}:")
            print(f"Ratio of Correct Predictions: {ratio_per_label[i]:.4f}")

            if len(average_ratio_per_label) <= i:
                average_ratio_per_label.append([ratio_per_label[i]])
            else:
                average_ratio_per_label[i].append(ratio_per_label[i])

        print("-" * 50)

        percentage_corrected = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
        percentage_corrected_labels.append(percentage_corrected)
average_ratio_per_label = np.mean(average_ratio_per_label, axis=1)

print("\nAverage Ratios Across All Patients:")
label_dict = {0: 'O', 1: 'N', 2: 'G', 3: 'P', 4: 'K', 5: 'B'}
for i, average_ratio in enumerate(average_ratio_per_label):
    print(f"Label {label_dict[i]}: {average_ratio:.4f}")