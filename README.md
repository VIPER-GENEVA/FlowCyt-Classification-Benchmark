# FlowCyt: A Comparative Study of Deep Learning Approaches for Multi-Class Classification in Flow Cytometry Benchmarking

This repository contains the benchmark dataset and code for the paper "FlowCyt: A Comparative Study of Deep Learning Approaches for Multi-Class Classification in Flow Cytometry" accepted at [CHIL@2024](https://chilconference.org/), [Track 2: Applications And Practice](https://chilconference.org/call-for-papers.html#tab-track-2) 
for benchmarking deep learning models in Medicine. Check out our latest works on the [VIPER Webpage](https://viper-geneva.github.io/)!

FlowCyt is the first comprehensive benchmark for evaluating multi-class single-cell classification methods on flow cytometry data. It comprises a richly annotated dataset of bone marrow samples from 30 patients, with ground truth labels for 5 important hematological cell types.

The goal is to facilitate standardized assessment and development of automated solutions for identifying cell populations, which can assist hematologists in analyzing these complex high-dimensional datasets.

## Repository Structure
The repository has the following structure:

- **data/**
  - **README.md**: Please refer to this for more explanations on data and graph generation.
  - **raw/**: Contains the original FCS files.
  - **data_original/**: Contains CSV data for each sample, saved as *Case_{i}.csv* with the six classes (A-population) of cells.
  - **data_original_sub/**: Contains CSV sub-population data for each sample, saved as *Case_{i}.csv* with the five classes (sub-population) of cells.

- **inductive/**
  - **README.md**: Please refer to this for inductive learning experiments and model reproducibility.
    
- **trans/**
  - **README.md**: Please refer to this for transductive learning experiments and model reproducibility.
 
- **utils/**
  - `train.py`: Training and testing function definitions, both for `gnn` and `dnn` models.
  - `weigths.py`: Class weights generation to take into account the strong class imbalance in the NLL loss function.

- **results/**: Saved results from experiments.

- **README.md**: This file.

- **requirements.txt**: Python package dependencies.

- `visualization/viz.py`: Visualization script to reproduce the [t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl) embedding, plot the feature importance, the degrees visualization and the attention explainer for `trans_gat.pt` models.

## Dataset
The dataset comprises 30 bone marrow samples with 14-dimensional flow cytometry measurements per cell, the relevant ones for reproducing the paper's experiment are the 12 selected in the corresponding code. Approximately 250,000 - 1,000,000 cells were measured per patient.

The data is stored in the `data/raw` folder as FCS files, the standard format output by flow cytometers.

Ground truth labels for 5 cell types are provided: T cells, B cells, monocytes, mast cells, and hematopoietic stem/progenitor cells (HSPCs). The labels are saved as separate FCS files per cell type in the `data/raw` folder.

The data has been anonymized. Please look at the paper for additional biological details about the samples and cell populations.

## Requirements

- Python 3.8 or later
- PyTorch 1.10.0 
- Torch Geometric 2.0.8 
- torchvision 0.11.1
- NumPy 1.21.2
- scikit-learn 0.24.2

## Installation
Installing via requirements:
```bash
pip install -r requirements.txt
````
Alternatively, you can install our `enviroment.yaml` file:
```bash
conda env create -f environment.yaml
conda activate flowcyt
```
Otherwise you may clone our repository:

```bash
git clone https://github.com/VIPER-GENEVA/FlowCyt-Classification-Benchmark.git
cd FlowCyt-Classification-Benchmark
```

## Quick Start
To reproduce paper's experiment, please run all the following command lines from this main project directory.

The following steps demonstrate how to run a GNN experiment:

1. Run one of these GNN models under the Inductive Learning framework:
```bash
python -u -m inductive.gnn_main --model GAT --num_layers 1 --hidden_features 16 --dropout 0.2 --in_heads 4 --out_heads 4 --input_dim 12 --output_dim 6 --max_num_epochs 1000 --start_lr 0.01 --num_repetitions 10
python -u -m inductive.gnn_main --model GCN --num_layers 1 --hidden_features 16 --dropout 0.3 --input_dim 12 --output_dim 6 --max_num_epochs 1000 --start_lr 0.01 --num_repetitions 10
python -u -m inductive.gnn_main --model SAGE --num_layers 1 --hidden_features 16 --dropout 0.3 --input_dim 12 --output_dim 6 --max_num_epochs 1000 --start_lr 0.01 --num_repetitions 10
```
This will train and evaluate a Graph Neural Network model using the default parameters. See `inductive/` for other available models.

2. Evaluate one of these GNN models under the Transductive Learning framework:
```bash
python -u -m trans.gnn_trans --model GAT --num_layers 1 --hidden_features 64 --dropout 0.2 --in_heads 2 --out_heads 2 --input_dim 12 --output_dim 6 --max_num_epochs 1000 --start_lr 0.01
python -u -m trans.gnn_trans --model GCN --num_layers 1 --hidden_features 64 --dropout 0.3 --input_dim 12 --output_dim 6 --max_num_epochs 1000 --start_lr 0.01
python -u -m trans.gnn_trans --model SAGE --num_layers 1 --hidden_features 64 --dropout 0.3 --input_dim 12 --output_dim 6 --max_num_epochs 1000 --start_lr 0.01
```

The script will print out performance metrics and also save predictions under `results/`. See `trans/` for details on specifying model hyperparameters and experiment configurations.

## Citation
If you find this benchmark dataset useful in your research, please cite the following paper:

```bash
@inproceedings{flowcyt2024,
  title={FlowCyt: A Comparative Study of Deep Learning Approaches for Multi-Class Classification in Flow Cytometry},
  author={Bini, Lorenzo and Nassajian Mojarrad, Fatemeh and Liarou, Margarita and Matthes, Thomas and Marchand-Maillet, Stéphane},
  booktitle={Conference on Health, Inference, and Learning (CHIL)},
  year={2024}
}
```

## Contact
Don't hesitate to get in touch with the authors with any questions or feedback about the benchmark. We are happy to receive suggestions for extensions and collaborations!
