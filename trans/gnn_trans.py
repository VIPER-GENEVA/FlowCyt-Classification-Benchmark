import argparse
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GATConv,SAGEConv, GCNConv
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import random
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.weights import get_class_weights_tensor
import sys

# Define parser
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
args = parser.parse_args()
print(sys.argv)

# Set seeds for reproducibility
seed_value = 77
torch.cuda.empty_cache()
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_GRAPH = 'data/sub_graph.pt' # data/A_graph.pt  
masked_graphs = torch.load(INPUT_GRAPH) 

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNNModel(input_dim=args.input_dim, output_dim=args.output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr, weight_decay=5e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=0.0000001)

# Add a new random mask to each graph  
for graph in masked_graphs:
  n = len(graph.y)
  mask = [random.random() < 0.5 for _ in range(n)]
  graph.rand_mask = torch.tensor(mask, dtype=torch.bool)
  
  # Mask 50% of nodes in train_mask, val_mask, test_mask
  #It's the same that doing it below inside the train, val loops
  #graph.train_mask = graph.train_mask & graph.rand_mask
  #graph.val_mask = graph.val_mask & graph.rand_mask
  #graph.test_mask = graph.test_mask & graph.rand_mask

best_val_loss = float('inf')
patience = 50 
counter = 0
class_weights_tensor = get_class_weights_tensor(INPUT_GRAPH)
class_weights_tensor = [weights.to(device) for weights in class_weights_tensor]

'''
The code belove masking 50% of the nodes inside the train_mask. 
It combines the train_mask with the rand_mask using the bitwise AND operator (&) 
to create the full_mask. The full_mask will only have True values for the nodes 
that are both in the train_mask and the rand_mask, effectively masking 50% of the nodes in the train_mask.
Same for val_mask and test_mask.
'''
for epoch in range(1, args.max_num_epochs + 1):
    
    torch.cuda.empty_cache()
    model.train()
    total_loss = 0
    for graph in masked_graphs:

        # Combine existing mask with new random mask for masking 50% of nodes
        full_mask = graph.train_mask & graph.rand_mask
        data = graph.to(device)
        optimizer.zero_grad()
        out = model(data)
        current_weights = class_weights_tensor[masked_graphs.index(graph)]
        criterion = nn.NLLLoss(weight=current_weights)
        loss = criterion(out[full_mask], data.y[full_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_train_loss = total_loss / len(masked_graphs)
    model.eval()
    val_loss = 0
    for graph in masked_graphs:
        # Combine existing mask with new random mask for masking 50% of nodes
        full_mask = graph.val_mask & graph.rand_mask
        data = graph.to(device)
        with torch.no_grad():
            out = model(data)
        current_weights = class_weights_tensor[masked_graphs.index(graph)]
        criterion = nn.NLLLoss(weight=current_weights)
        loss = criterion(out[full_mask], data.y[full_mask])
        val_loss += loss.item()

    average_val_loss = val_loss / len(masked_graphs)
    print(f'Epoch: {epoch}, Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}')
    scheduler.step(average_val_loss)
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping after {epoch} epochs.')
            break

print('Training and validation completed.')

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
model.eval()
for graph in masked_graphs:
    full_mask = graph.test_mask & graph.rand_mask
    data = graph.to(device)
    with torch.no_grad():
        out = model(data)
    
    predicted_labels = torch.argmax(out[full_mask], dim=1)
    true_labels = data.y[full_mask]
    
    accuracy = accuracy_score(true_labels.cpu(), predicted_labels.cpu())
    precision = precision_score(true_labels.cpu(), predicted_labels.cpu(), average='weighted', zero_division=1)
    recall = recall_score(true_labels.cpu(), predicted_labels.cpu(), average='weighted', zero_division=1)
    f1 = f1_score(true_labels.cpu(), predicted_labels.cpu(), average='weighted', zero_division=1)
    
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

for i in range(len(masked_graphs)):
    print(f'Graph {i+1}:')
    print(f'Accuracy: {accuracy_scores[i]:.4f}')
    print(f'Precision: {precision_scores[i]:.4f}')
    print(f'Recall: {recall_scores[i]:.4f}')
    print(f'F1 Score: {f1_scores[i]:.4f}')
    print()

print('Testing completed.')
average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
average_precision = sum(precision_scores) / len(precision_scores)
average_recall = sum(recall_scores) / len(recall_scores)
average_f1 = sum(f1_scores) / len(f1_scores)

print(f'Average Accuracy: {average_accuracy:.4f}')
print(f'Average Precision: {average_precision:.4f}')
print(f'Average Recall: {average_recall:.4f}')
print(f'Average F1 Score: {average_f1:.4f}')
