#! /bin/bash
#
#SBATCH --account aauhpc_slim     # account
#SBATCH --nodes 1 		 # number of nodes
#SBATCH --gres=gpu:2             # Number of GPUs per node
#SBATCH --time 24:00:00          # max time (HH:MM:SS)
#SBATCH --cpus-per-task=24       # run two processes on the same node

echo Running on "$(hostname)"
echo Available nodes: "$SLURM_NODELIST"
echo Slurm_submit_dir: "$SLURM_SUBMIT_DIR"
echo Start time: "$(date)"

# Load the Python environment
# module purge
# module use /work/aauhpc/software/modules/all/
# module add python-intel/3.5.2 CUDA/9.0.176 cuDNN/7.1-CUDA-9.0.176
# source activate /work/aauhpc/jilin/tensorflow/
# conda activate tensorflow  
# Start your python application

#module load intel/2018.05
#module load openmpi/3.0.2

#python baseline_train_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --base_line 'gp' &
#python baseline_train_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --base_line 'svr' &
#python baseline_train_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --base_line 'var' &
#
##python baseline_train_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 3 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --base_line 'svr' &
#python baseline_train_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 3 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --base_line 'var' &
python baseline_train_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 3 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --base_line 'gp' &

wait


