#!/bin/sh
#SBATCH --job-name flowcyt         
#SBATCH --error run.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output run.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    
#SBATCH --cpus-per-task 4            
#SBATCH --mem 32GB                   
#SBATCH --partition debug-gpu         
#SBATCH --gres=gpu:1 #,VramPerGpu:15G
#SBATCH --time 0-00:14:59                  

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate flowcyt
###################################
export CUDA_VISIBLE_DEVICES=0 # GPU devices 0,1,2...

nvidia-smi
#### INDUCTIVE
python -u -m inductive.gnn_main --model GAT --num_layers 1 --hidden_features 16 --dropout 0.2 --in_heads 4 --out_heads 4 --input_dim 12 --output_dim 6 --max_num_epochs 1000 --start_lr 0.01 --num_repetitions 10
#python -u -m inductive.dnn_main --num_layers 5 --hidden_dim 256 --input_dim 12 --output_dim 6 --dropout 0.3 --max_num_epochs 300 --batch_size 1 --start_lr 0.01 --num_repetitions 10
#python -u -m inductive.gaussian_main --max_num_epochs 100 --batch_size 1 --n_components 6 --max_iter 1000 --num_repetitions 4   
#python -u -m inductive.rf_main --max_num_epochs 100 --batch_size 1 --num_repetitions 5 --n_estimators 10 --max_depth 3
#python -u -m inductive.xgb_main --max_num_epochs 100 --batch_size 1 --num_repetitions 5 --n_estimators 10 --max_depth 3 --learning_rate 0.01

#### TRANS
#python -u -m trans.gnn_trans --model GAT --num_layers 1 --hidden_features 64 --dropout 0.3 --in_heads 2 --out_heads 2 --input_dim 12 --output_dim 6 --max_num_epochs 1000 --start_lr 0.01

#### DATA_GEN
#python -u -m data.A_generation
#python -u -m data.graph_generation --in_dir data/data_original --out_dir data/A_graph.pt
