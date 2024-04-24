import argparse
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Dataset
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import random
import torch
from sklearn.mixture import GaussianMixture
import sys
import os

# Set seeds for reproducibility
seed_value = 77
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Define parser
parser = argparse.ArgumentParser(description='Gaussian Mixture Models Evaluation Script')
parser.add_argument('--max_num_epochs', type=int, default=200, help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for DataLoader')
parser.add_argument('--n_components', type=int, default=6, help='Number of trees in the forest')
parser.add_argument('--max_iter', type=int, default=1000, help='Number of iterations for gaussian mixture')
parser.add_argument('--num_repetitions', type=int, default=10, help='Number of repetitions for cross validation')
args = parser.parse_args()
print(sys.argv)

PRINT_MEMORY = False
device_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)
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
    df = pd.read_csv(f"{INPUT_FOLDER}/Case_{j+1}.csv")  
    
    label0_count.append(len(df))

class MyGraphDataset(Dataset):
    def __init__(self,  num_samples,transform=None, pre_transform=None):
        super(MyGraphDataset, self).__init__(transform, pre_transform)
        self.num_samples = num_samples
        self.data_list = torch.load(INPUT_GRAPH)  

    def len(self):
        return self.num_samples

    def get(self, idx):
        return self.data_list[idx]

def test(loader, clf):
    all_preds = []
    all_labels = []

    for data in loader:
        features = data.x
        labels = data.y.numpy()
        preds = clf.predict(features)  
        all_preds.extend(preds)
        all_labels.extend(labels)

    accuracy = accuracy_score(all_labels, all_preds)  
    return accuracy

clf = GaussianMixture(
    n_components=args.n_components,  
    covariance_type='full',
    random_state=77,
    max_iter=args.max_iter  # Increase the max_iter parameter
)

def gaussian_evaluation(clf, max_num_epochs=200, batch_size=128, num_repetitions=10):
    dataset = MyGraphDataset(num_samples=len(torch.load(INPUT_GRAPH))).shuffle()  
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

            best_val_acc = 0.0

            early_stopping_counter = 0
            early_stopping_threshold = 15  # Number of epochs without improvement to trigger early stopping

            for epoch in range(1, max_num_epochs + 1):
                for data in train_loader:
                    data = data.to(device)
                    features = data.x.cpu().numpy()
                    labels = data.y.cpu().numpy()
                    clf.fit(features, labels)  

                val_acc = test(val_loader, clf)  

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_acc = test(test_loader, clf) * 100.0
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= early_stopping_threshold:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            for data in test_loader:
                data = data.to(device)
                features = data.x.cpu().numpy()
                labelss = data.y.cpu().numpy()
                predss = clf.predict(features) 
                idx = label0_count.index(len(labelss)) + 1
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
patient_dict = gaussian_evaluation(clf, max_num_epochs=args.max_num_epochs, batch_size=args.batch_size, num_repetitions=args.num_repetitions)

# Initialize a list to store the ratios for each label across all patients
average_ratio_per_label = []
percentage_corrected_labels = []
results_dir = "res_ind_gaussian"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
results_file = os.path.join(results_dir, "res_ind_gaussian.txt")
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
