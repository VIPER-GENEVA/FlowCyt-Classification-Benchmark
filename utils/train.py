import torch
import torch.nn as nn

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.empty_cache()

with ClearCache():
    # One training epoch for GNN model.
    def train_gnn(train_loader, model, optimizer, device, class_weights_tensor):
        model.train()
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            # Apply combined mask
            current_weights = class_weights_tensor[batch_idx]
            criterion = nn.NLLLoss(weight=current_weights)
            #criterion = nn.CrossEntropyLoss(weight=current_weights)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
        #return output

# Get acc. of GNN model.
def test_gnn(loader, model, device):
    with torch.no_grad():
        model.eval()
        correct = 0
        for data in loader:
            data = data.to(device)
            output = model(data)
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()/len(pred)
        return correct / len(loader.dataset)
    
with ClearCache():
    # One training epoch for DNN model.
    def train_dnn(train_loader, model, optimizer, device, class_weights_tensor):
        model.train()
        for batch_idx, data in enumerate(train_loader):
            matrix = data.x.to(device)
            labels = data.y.to(device)
            optimizer.zero_grad()
            output = model(matrix)
            # Apply combined mask
            current_weights = class_weights_tensor[batch_idx]
            criterion = nn.NLLLoss(weight=current_weights)
            #criterion = nn.CrossEntropyLoss(weight=current_weights)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

# Get acc. of DNN model.
def test_dnn(loader, model, device):
    with torch.no_grad():
        model.eval()
        correct = 0
        for data in loader:
            matrix = data.x.to(device)
            labels = data.y.to(device)
            output = model(matrix)
            pred = output.max(dim=1)[1]
            correct += pred.eq(labels).sum().item()/len(pred)
        return correct / len(loader.dataset)