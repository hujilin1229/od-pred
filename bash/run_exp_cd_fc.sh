#! /bin/bash
#
#SBATCH --account aauhpc_slim     # account
#SBATCH --nodes 1 		 # number of nodes
#SBATCH --gres=gpu:2             # Number of GPUs per node
#SBATCH --time 24:00:00          # max time (HH:MM:SS)
#SBATCH --cpus-per-task=12       # run two processes on the same node

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

module load intel/2018.05
module load openmpi/3.0.2

# srun -n 1 --exclusive python mgrnn_train.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --num_gpus 2 --fill_mean true --learning_rate 0.001 & 
# srun -n 1 --exclusive python mgrnn_train_hist.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --num_gpus 2 --fill_mean false --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 10 --batch_size 32 --loss_func 'EMD' & 
# srun -n 1 --exclusive python mgrnn_train_hist.py --test_every_n_epochs 10 --sample_rate 20 --data_format 'speed' --num_gpus 2 --fill_mean=false --sparse_removal=false --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 16 --loss_func 'L2' --activate_func 'tanh' --pool_type '_mpool' & 
 
#python fcgrnn_train_cd.py --test_every_n_epochs 1 --sample_rate 15 --data_format 'speed' \
#--seq_len 3 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 \
#--lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --drop_out 0.2 --use_curriculum_learning=True --sigma 9 --hopk 4 \
#--optimizer 'adam' --fc_method='od' --trace_ratio 0.00001 &

# srun -n 1 --exclusive python fcgrnn_train_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --drop_out 0.2 --use_curriculum_learning=True --sigma 9 --hopk 4 --optimizer 'adam' --fc_method='od' --trace_ratio 0.000001&
# srun -n 1 --exclusive python fcgrnn_train_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --drop_out 0.2 --use_curriculum_learning=True --sigma 9 --hopk 4 --optimizer 'adam' --fc_method='od' --trace_ratio 0.0001&
# srun -n 1 --exclusive python fcgrnn_train_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --drop_out 0.2 --use_curriculum_learning=True --sigma 9 --hopk 4 --optimizer 'adam' --fc_method='od' --trace_ratio 0.001 &

python fcgrnn_train_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 \
--num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 \
--batch_size 8 --loss_func 'L2Norm' --drop_out 0.2 --use_curriculum_learning=True --sigma 9 --hopk 4 --optimizer 'Adam' \
--fc_method='od' --trace_ratio 0.000001


# srun -n 1 --exclusive python fcgrnn_train_cd.py --test_every_n_epochs 1 --sample_rate 15 --data_format 'speed' --seq_len 3 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2' --drop_out 0.2 --use_curriculum_learning=True --sigma 9 --hopk 4 --optimizer 'adam' --fc_method='direct' &
# srun -n 1 --exclusive python mgrnn_train_hist.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=True --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'EMD' --activate_func 'tanh' --pool_type '_mpool' --drop_out 0.2 --use_curriculum_learning=True --sigma 9 --hopk 2 --optimizer 'ada' &
# srun -n 1 --exclusive python mgrnn_train_hist.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 16 --loss_func 'L2' --activate_func 'tanh' --pool_type '_mpool' --drop_out 0.2 --use_curriculum_learning=True &
#srun -n 1 --exclusive python mgrnn_train_hist.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 5 --horizon 3 --num_gpus 2 --fill_mean=True --sparse_removal=false --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 4 --loss_func 'L2' --activate_func 'tanh' --pool_type '_mpool' & 
#srun -n 1 --exclusive python mgrnn_train_hist.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 5 --horizon 3 --num_gpus 2 --fill_mean=True --sparse_removal=false --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 4 --loss_func 'EMD' --activate_func 'tanh' --pool_type '_mpool' & 
#srun -n 1 --exclusive python mgrnn_train_hist.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 5 --horizon 3 --num_gpus 2 --fill_mean=True --sparse_removal=false --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 4 --loss_func 'KL' --activate_func 'tanh' --pool_type '_mpool' & 
#srun -n 1 --exclusive python mgrnn_train_hist.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 5 --horizon 3 --num_gpus 2 --fill_mean=True --sparse_removal=false --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 4 --loss_func 'KL' --activate_func 'tanh' --pool_type '_mpool' & 
#srun -n 1 --exclusive python mgrnn_train_hist.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 5 --horizon 3 --num_gpus 2 --fill_mean=True --sparse_removal=false --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 4 --loss_func 'KL' --activate_func 'tanh' --pool_type '_mpool' & 
# srun -n 1 --exclusive python mgrnn_train_hist.py --test_every_n_epochs 10 --sample_rate 30 --data_format 'speed' --num_gpus 2 --fill_mean=false --sparse_removal=false --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 10 --batch_size 16 --loss_func 'L2' --activate_func 'tanh' --pool_type '_mpool' & 

# nvidia-smi
# export CUDA_VISIBLE_DEVICE=0
# export CUDA_VISIBLE_DEVICE=1
# srun -n 1 --exclusive python mgrnn_train.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'duration'
#-srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "chengdu" --mode "prediction" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[8, 4]" --conv_size "[4, 4]" --learning_rate 0.004 --regularization 0.006 --dropout 0.27 --decay_rate 0.98 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "server_kdd" --mode "prediction" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[8, 8]" --learning_rate 0.003 --regularization 0.08 --dropout 0.14 --decay_rate 0.98 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "chengdu" --mode "estimation" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[8, 4]" --conv_size "[4, 4]" --learning_rate 0.005 --regularization 0.044 --dropout 0.12 --decay_rate 1.0 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "server_kdd" --mode "estimation" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[8, 8]" --learning_rate 0.0001 --regularization 0.001 --dropout 0.4 --decay_rate 0.99 &

## hist no_embed
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "chengdu" --mode "prediction" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[8, 4]" --conv_size "[4, 4]" --learning_rate 0.0065 --regularization 0.1 --dropout 0.6 --decay_rate 1.0 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "server_kdd" --mode "prediction" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[8, 8]" --learning_rate 5.6e-5 --regularization 0.001 --dropout 0.0 --decay_rate 1.0 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "chengdu" --mode "estimation" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[8, 4]" --conv_size "[4, 4]" --learning_rate 0.077 --regularization 0.1 --dropout 0.006 --decay_rate 0.94 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "server_kdd" --mode "estimation" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[8, 8]" --learning_rate 0.1 --regularization 0.001 --dropout 0.0 --decay_rate 0.9 &

# avg no_emd
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "chengdu" --mode "prediction" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[8, 4]" --conv_size "[4, 4]" --learning_rate 0.0265 --regularization 0.1 --dropout 0.0 --decay_rate 0.9 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "server_kdd" --mode "prediction" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[8, 8]" --learning_rate 0.01 --regularization 0.006 --dropout 0.19 --decay_rate 0.93 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "chengdu" --mode "estimation" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[8, 4]" --conv_size "[4, 4]" --learning_rate 7.55e-5 --regularization 0.01 --dropout 0.033 --decay_rate 1.0 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "server_kdd" --mode "estimation" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[8, 8]" --learning_rate 0.0005 --regularization 0.08 --dropout 0.2 --decay_rate 0.94 &
#
## hist embed
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "chengdu" --mode "prediction" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[8, 4]" --conv_size "[4, 4]" --learning_rate 0.005 --regularization 0.1 --dropout 0.0 --decay_rate 1.0 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "server_kdd" --mode "prediction" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[8, 8]" --learning_rate 0.002 --regularization 0.008 --dropout 0.2 --decay_rate 0.99 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "chengdu" --mode "estimation" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[8, 4]" --conv_size "[4, 4]" --learning_rate 0.008 --regularization 0.001 --dropout 0.6 --decay_rate 0.9 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "server_kdd" --mode "estimation" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[8, 8]" --learning_rate 0.0023 --regularization 0.003 --dropout 0.045 --decay_rate 0.96 &

# avg embed
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "chengdu" --mode "prediction" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[8, 4]" --conv_size "[4, 4]" --learning_rate 0.018 --regularization 0.05 --dropout 0.2 --decay_rate 0.95 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "server_kdd" --mode "prediction" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[8, 8]" --learning_rate 0.001 --regularization 0.1 --dropout 0.6 --decay_rate 1.0 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "chengdu" --mode "estimation" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[8, 4]" --conv_size "[4, 4]" --learning_rate 0.476 --regularization 0.035 --dropout 0.33 --decay_rate 0.97 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "server_kdd" --mode "estimation" --conv "rnn" --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[8, 8]" --learning_rate 0.1 --regularization 0.001 --dropout 0.0 --decay_rate 0.9 &

### hist random prediction
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "chengdu" --mode "prediction" --conv "cnn" --filter 'conv1' --pos_embed False --context_embed False --is_coarsen False --stop_early True --num_epochs 201 --num_kernels "[8, 4]" --conv_size "[4, 4]" --learning_rate 0.1 --regularization 0.001 --dropout 0.0 --decay_rate 0.9 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "server_kdd" --mode "prediction" --conv "cnn" --filter 'conv1' --pos_embed False --context_embed False --is_coarsen False --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[8, 8]" --learning_rate 0.0018 --regularization 0.0097 --dropout 0.027 --decay_rate 0.998 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "chengdu" --mode "prediction" --conv "gcnn" --filter 'chebyshev5' --pos_embed False --context_embed False --is_coarsen True --stop_early True --num_epochs 201 --num_kernels "[8, 4]" --conv_size "[4, 4]" --learning_rate 0.01 --regularization 0.001 --dropout 0.0 --decay_rate 0.9 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "server_kdd" --mode "prediction" --conv "gcnn" --filter 'chebyshev5' --pos_embed False --context_embed False --is_coarsen True --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[8, 8]" --learning_rate 0.0046 --regularization 0.004 --dropout 0.0 --decay_rate 0.9 &

#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "chengdu" --mode "prediction" --conv "gcnn" --filter 'chebyshev5' --pos_embed False --context_embed False --is_coarsen True --stop_early True --num_epochs 201 --num_kernels "[8, 8]" --conv_size "[8, 4]" --pool_size "[2, 2]" --learning_rate 0.001 --regularization 0.001 --dropout 0.0 --decay_rate 0.935 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "server_kdd" --mode "prediction" --conv "gcnn" --filter 'chebyshev5' --pos_embed False --context_embed False --is_coarsen True --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[8, 8]" --pool_size "[4, 2]" --learning_rate 0.0008 --regularization 0.001 --dropout 0.0 --decay_rate 1.0 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "chengdu" --mode "prediction" --conv "gcnn" --filter 'chebyshev5' --pos_embed True --context_embed True --is_coarsen True --stop_early True --num_epochs 201 --num_kernels "[8, 8]" --conv_size "[8, 4]" --pool_size "[2, 2]" --learning_rate 0.0014 --regularization 0.001 --dropout 0.0 --decay_rate 0.9 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "server_kdd" --mode "prediction" --conv "gcnn" --filter 'chebyshev5' --pos_embed True --context_embed True --is_coarsen True --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[8, 8]" --pool_size "[4, 2]" --learning_rate 0.000684 --regularization 0.001 --dropout 0.04 --decay_rate 0.997 &

#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "chengdu" --mode "estimation" --target 'avg' --classif_loss 'mape' --conv "rnn" --filter 'chebyshev5' --pos_embed False --context_embed True --is_coarsen False --stop_early True --num_epochs 201 --num_kernels "[8, 8]" --conv_size "[8, 4]" --pool_size "[2, 2]" --learning_rate 0.013 --regularization 0.1 --dropout 0.6 --decay_rate 0.9 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "server_kdd" --mode "estimation" --target 'avg' --classif_loss 'mape' --conv "rnn" --filter 'chebyshev5' --pos_embed False --context_embed True --is_coarsen False --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[8, 8]" --pool_size "[4, 2]" --learning_rate 0.1 --regularization 0.1 --dropout 0.0 --decay_rate 0.9 &

#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "server_kdd" --mode "prediction" --target 'hist' --classif_loss 'kl' --test True --conv "cnn" --filter 'conv1' --pos_embed False --context_embed False --is_coarsen False --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[4, 4]" --pool_size "[2, 2]" --learning_rate 0.004 --regularization 0.002 --dropout 0.5 --decay_rate 0.95 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "server_kdd" --mode "prediction" --target 'hist' --classif_loss 'kl' --test True --conv "rnn" --filter 'chebyshev5' --pos_embed False --context_embed True --is_coarsen False --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[4, 4]" --pool_size "[2, 2]" --learning_rate 0.000087 --regularization 0.1 --dropout 0.27 --decay_rate 1.0 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "server_kdd" --mode "prediction" --target 'hist' --classif_loss 'kl' --test True --conv "gcnn" --filter 'chebyshev5' --pos_embed False --context_embed False --is_coarsen True --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[4, 4]" --pool_size "[2, 2]" --learning_rate 0.003 --regularization 0.01 --dropout 0.0 --decay_rate 0.92 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "server_kdd" --mode "prediction" --target 'hist' --classif_loss 'kl' --test True --conv "gcnn" --filter 'chebyshev5' --pos_embed True --context_embed True --is_coarsen True --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[4, 4]" --pool_size "[2, 2]" --learning_rate 0.0034 --regularization 0.046 --dropout 0.1 --decay_rate 0.97 &

#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "server_kdd" --mode "prediction" --target 'hist' --classif_loss 'kl' --test True --conv "cnn" --filter 'conv1' --pos_embed False --context_embed False --is_coarsen False --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[4, 4]" --pool_size "[4, 4]" --learning_rate 0.0175 --regularization 0.001 --dropout 0.0 --decay_rate 0.98 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --server_name "server_kdd" --mode "prediction" --target 'hist' --classif_loss 'kl' --test True --conv "rnn" --filter 'chebyshev5' --pos_embed False --context_embed True --is_coarsen False --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[4, 4]" --pool_size "[4, 4]" --learning_rate 0.032 --regularization 0.01 --dropout 0.14 --decay_rate 0.95 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "server_kdd" --mode "prediction" --target 'hist' --classif_loss 'kl' --test True --conv "gcnn" --filter 'chebyshev5' --pos_embed False --context_embed False --is_coarsen True --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[4, 4]" --pool_size "[4, 4]" --learning_rate 0.0128 --regularization 0.001 --dropout 0.0 --decay_rate 0.96 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --server_name "server_kdd" --mode "prediction" --target 'hist' --classif_loss 'kl' --test True --conv "gcnn" --filter 'chebyshev5' --pos_embed True --context_embed True --is_coarsen True --stop_early True --num_epochs 201 --num_kernels "[16, 16]" --conv_size "[4, 4]" --pool_size "[4, 4]" --learning_rate 0.0026 --regularization 0.023 --dropout 0.075 --decay_rate 0.98 &
#


#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 0 --test True --server_name "server_kdd" --mode "prediction" --conv "gcnn" --filter 'chebyshev5' --pos_embed True --context_embed True --is_coarsen True --stop_early True --num_epochs 101 --num_kernels "[16, 16]" --conv_size "[8, 8]" --pool_size "[4, 2]" --learning_rate 0.00064 --regularization 0.003 --dropout 0.4 --decay_rate 1.0 &
#srun -n 1 --exclusive python gcrn_main_gcnn.py --gpu_id 1 --test True --server_name "chengdu" --mode "prediction" --conv "gcnn" --filter 'chebyshev5' --pos_embed True --context_embed True --is_coarsen True --stop_early True --num_epochs 101 --num_kernels "[8, 8]" --conv_size "[8, 4]" --pool_size "[2, 2]" --learning_rate 0.0077 --regularization 0.001 --dropout 0.45 --decay_rate 0.9 &



wait


