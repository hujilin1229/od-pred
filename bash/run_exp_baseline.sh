#! /bin/bash
#
#SBATCH --account aauhpc_slim     # account
#SBATCH --nodes 1 		 # number of nodes
#SBATCH --gres=gpu:2             # Number of GPUs per node
#SBATCH --time 24:00:00          # max time (HH:MM:SS)
#SBATCH --cpus-per-task=24       # run two processes on the same node


python baseline_train.py --test_every_n_epochs 1 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --fill_mean=False --sparse_removal=False --base_line 'gp'
#python baseline_train.py --test_every_n_epochs 1 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --fill_mean=False --sparse_removal=False --base_line 'var'

python baseline_train.py --test_every_n_epochs 1 --sample_rate 15 --data_format 'speed' --seq_len 3 --horizon 3 --fill_mean=Fasle --sparse_removal=False --base_line 'gp'
#python baseline_train.py --test_every_n_epochs 1 --sample_rate 15 --data_format 'speed' --seq_len 3 --horizon 3 --fill_mean=False --sparse_removal=False --base_line 'var'



