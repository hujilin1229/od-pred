#! /bin/bash
##
##SBATCH --account aauhpc_fat     # account
##SBATCH --nodes 1 		 # number of nodes
##SBATCH --gres=gpu:2             # Number of GPUs per node
##SBATCH --time 24:00:00          # max time (HH:MM:SS)
##SBATCH --cpus-per-task=6       # run two processes on the same node
#
#echo Running on "$(hostname)"
#echo Available nodes: "$SLURM_NODELIST"
#echo Slurm_submit_dir: "$SLURM_SUBMIT_DIR"
#echo Start time: "$(date)"
#
## Load the Python environment
## module purge
## module use /work/aauhpc/software/modules/all/
## module add python-intel/3.5.2 CUDA/9.0.176 cuDNN/7.1-CUDA-9.0.176
## source activate /work/aauhpc/jilin/tensorflow/
## conda activate tensorflow
## Start your python application
#
#module load intel/2018.05
#module load openmpi/3.0.2
 

#python fcgrnn_train_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' \
#--seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 \
#--lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'tanh' --pool_type '_mpool' --drop_out 0.4 \
#--use_curriculum_learning=True --sigma 9 --hopk 4 --optimizer 'adam' --fc_method='od' --trace_ratio 0.1 --is_restore=True \
#--model_dir='./logs/fcrnn_L_h_3_FC3-FC3_lr_0.001_bs_8_d_0.2_sl_6_L2Norm_polygon_speed_15_1101143905_od_0.0' --model_filename='models-0.1711-15820' &
# srun -n 1 --exclusive python fcgrnn_train_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2' --activate_func 'tanh' --pool_type '_mpool' --drop_out 0.4 --use_curriculum_learning=True --sigma 9 --hopk 4 --optimizer 'adam' --fc_method='direct' --trace_ratio 0.1 --is_restore=True --model_dir='./logs/fcrnn_L_h_3_FC3-FC3_lr_0.001_bs_8_d_0.4_sl_6_L2_polygon_speed_15_1029181023_direct_0.0/' --model_filename='models-0.1611-9492' &

# New Dataset
# rnn units = [3, 3], latent dim 5 S6H3
#python fcgrnn_train_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' \
# --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 \
# --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'tanh' --pool_type '_mpool' --drop_out 0.4 \
# --use_curriculum_learning=True --sigma 9 --hopk 4 --optimizer 'adam' --fc_method='od' --trace_ratio 0.1 \
# --is_restore=True --model_dir='./logs/fcrnn_L_h_3_FC3-FC3_lr_0.001_bs_8_d_0.2_sl_6_L2Norm_polygon_speed_15_1101203053_od_1e-06' \
# --model_filename='models-0.1657-23730' &

# Effective OD, S6H3
#python fcgrnn_train_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 \
#--num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 \
#--batch_size 8 --loss_func 'L2Norm' --drop_out 0.2 --use_curriculum_learning=True --sigma 9 --hopk 4 --optimizer 'Adam' \
#--fc_method='od' --trace_ratio 0.000001 \
#--is_restore=True --model_dir='./logs/fcrnn_L_h_3_FC2-FC2_lr_0.001_bs_8_d_0.2_sl_6_L2Norm_polygon_speed_15_0122110648_od_1e-06' \
#--model_filename='models-0.1240-19210' &

#python fcgrnn_train_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' \
# --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 \
# --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2' --activate_func 'tanh' --pool_type '_mpool' --drop_out 0.4 \
# --use_curriculum_learning=True --sigma 9 --hopk 4 --optimizer 'adam' --fc_method='direct' --trace_ratio 0.1 \
# --is_restore=True --model_dir='./logs/fcrnn_L_h_3_FC3-FC3_lr_0.001_bs_8_d_0.2_sl_6_L2_polygon_speed_15_1101203219_direct_0.0' \
# --model_filename='models-0.1395-40906' &

# S3H3
#python fcgrnn_train_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' \
# --seq_len 3 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 \
# --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2' --activate_func 'tanh' --pool_type '_mpool' --drop_out 0.4 \
# --use_curriculum_learning=True --sigma 9 --hopk 4 --optimizer 'adam' --fc_method='direct' --trace_ratio 0.1 \
# --is_restore=True --model_dir='./logs/fcrnn_L_h_3_FC3-FC3_lr_0.001_bs_8_d_0.2_sl_3_L2_polygon_speed_15_1105210842_direct_0.0' \
# --model_filename='models-0.1391-34579' &

python fcgrnn_train_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' \
--seq_len 3 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 \
--lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'tanh' --pool_type '_mpool' --drop_out 0.4 \
--use_curriculum_learning=True --sigma 9 --hopk 4 --optimizer 'adam' --fc_method='od' --trace_ratio 0.00001 \
--is_restore=True --model_dir='./logs/fcrnn_L_h_3_FC2-FC2_lr_0.001_bs_8_d_0.2_sl_3_L2Norm_polygon_speed_15_1105211013_od_1e-05' \
--model_filename='models-0.1263-14427' &



wait


