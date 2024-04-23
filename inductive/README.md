# Inductive Learning Experiments

This directory contains code for running inductive learning experiments on the FlowCyt benchmark dataset.

Inductive learning refers to the standard supervised machine learning pipeline, where models are trained on labeled examples and evaluated on held-out test data.

## Models

The following supervised learning models are implemented under `models/`:

- `gnn_main.py`: Graph Neural Networks
- `dnn_main.py`: Fully-connected Deep Neural Networks
- `rf.py`: Random Forest 
- `xgb_main.py`: XGBoost 
- `gaussian_main.py`: Gaussian Mixture Model

## Experiments
To reproduce the paper's experiments please run all the following command lines from the main project directory.
To run a machine learning experiment, use:

```bash
python -u -m inductive.gaussian_main --max_num_epochs 100 --batch_size 1 --n_components 6 --max_iter 1000 --num_repetitions 4   
python -u -m inductive.rf_main --max_num_epochs 100 --batch_size 1 --num_repetitions 5 --n_estimators 10 --max_depth 3
python -u -m inductive.xgb_main --max_num_epochs 100 --batch_size 1 --num_repetitions 5 --n_estimators 10 --max_depth 3 --learning_rate 0.01 
```

For example, this will train test, and evaluate XGBoost and Random Forest models with 10 estimators on the benchmark dataset. Users are free to choose on which populations (total or sub) run the experiments, simply by modifying `INPUT_FOLDER` and `INPUT_GRAPH` inside the `gaussian_main.py`, `rf_main.py` and `xgb_main.py` scripts.

To run a deep learning experiment either use (early_stopping is set to 50 by default):

```bash
python -u -m inductive.gnn_main --model GAT --num_layers 1 --hidden_features 16 --dropout 0.2 --in_heads 4 --out_heads 4 --input_dim 12 --output_dim 6 --max_num_epochs 1000 --start_lr 0.01 --num_repetitions 10
python -u -m inductive.gnn_main --model GCN --num_layers 1 --hidden_features 16 --dropout 0.3 --input_dim 12 --output_dim 6 --max_num_epochs 1000 --start_lr 0.01 --num_repetitions 10
python -u -m inductive.gnn_main --model SAGE --num_layers 1 --hidden_features 16 --dropout 0.3 --input_dim 12 --output_dim 6 --max_num_epochs 1000 --start_lr 0.01 --num_repetitions 10
```

or

```bash
python -u -m inductive.dnn_main --num_layers 5 --hidden_dim 256 --input_dim 12 --output_dim 6 --dropout 0.3 --max_num_epochs 300 --batch_size 1 --start_lr 0.01 --num_repetitions 10
```

Both `gnn_main.py` and `dnn_main` call the utils functions for training, validation, and testing inside the `utils/` folder. 

For each model `input_dim` is always set to 12, and `ouput_dim` may vary from 5 to 6 depending on using sub/total populations the users may choose (simply need to modify `INPUT_GRAPH` and `INPUT_FOLDER` inside `gnn_main.py` and `dnn_main.py`).

Results are saved to `results/` including per-class metrics.

## Citation

If you find this useful for your research, please cite the FlowCyt paper:

```
@inproceedings{flowcyt2024,
  title={FlowCyt: A Comparative Study of Deep Learning Approaches for Multi-Class Classification in Flow Cytometry},
  author={Bini, Lorenzo and Nassajian Mojarrad, Fatemeh and Liarou, Margarita and Matthes, Thomas and Marchand-Maillet, Stéphane},
  booktitle={Conference on Health, Inference, and Learning (CHIL)},
  year={2024}
}
```
