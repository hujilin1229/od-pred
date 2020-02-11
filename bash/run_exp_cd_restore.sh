#! /bin/bash
#
#SBATCH --account aauhpc_slim     # account
#SBATCH --nodes 1 		 # number of nodes
#SBATCH --gres=gpu:2             # Number of GPUs per node
#SBATCH --time 24:00:00          # max time (HH:MM:SS)
#SBATCH --cpus-per-task=24       # run two processes on the same node

#echo Running on "$(hostname)"
#echo Available nodes: "$SLURM_NODELIST"
#echo Slurm_submit_dir: "$SLURM_SUBMIT_DIR"
#echo Start time: "$(date)"

# Load the Python environment
# module purge
# module use /work/aauhpc/software/modules/all/
# module add python-intel/3.5.2 CUDA/9.0.176 cuDNN/7.1-CUDA-9.0.176
# source activate /work/aauhpc/jilin/tensorflow/
# conda activate tensorflow  
# Start your python application

#module load intel/2018.05
#module load openmpi/3.0.2

# srun -n 1 --exclusive python mgrnn_train.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --num_gpus 2 --fill_mean true --learning_rate 0.001 & 
# srun -n 1 --exclusive python mgrnn_train_hist.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --num_gpus 2 --fill_mean false --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 10 --batch_size 32 --loss_func 'EMD' & 
# srun -n 1 --exclusive python mgrnn_train_hist.py --test_every_n_epochs 10 --sample_rate 20 --data_format 'speed' --num_gpus 2 --fill_mean=false --sparse_removal=false --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 16 --loss_func 'L2' --activate_func 'tanh' --pool_type '_mpool' & 

# New Data Set S6H3
#python mgrnn_train_hist_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 \
#--num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 \
#--loss_func 'L2Norm' --activate_func 'tanh' --pool_type '_apool' --drop_out 0.2 --use_curriculum_learning=True --sigma 9 \
#--hopk 2 --optimizer 'adam' --shuffle_training=True --trace_ratio 0.0001 --is_restore=True \
#--model_dir './logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_6_L2Norm_polygon_speed_15_1101202342_False_sigma9_hopk2_adam_tanh__apool_0.0001/' \
#--model_filename 'models-0.1355-6102' &

# S3H3
#python mgrnn_train_hist_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' \
#--seq_len 3 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 \
#--lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'tanh' --pool_type '_apool' --drop_out 0.2 \
#--use_curriculum_learning=True --sigma 9 --hopk 2 --optimizer 'adam' --shuffle_training=True --trace_ratio 0.0001 \
#--is_restore=True --model_dir './logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_3_L2Norm_polygon_speed_15_1105212839_False_sigma9_hopk2_adam_tanh__apool_0.0001/' \
#--model_filename 'models-0.1360-6412' &


# Experiments for different sigma and alpha
#python mgrnn_train_hist_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'tanh' --pool_type '_apool' --drop_out 0.2 --use_curriculum_learning=True --sigma 9 --hopk 4 --optimizer 'adam' --shuffle_training=True --trace_ratio 0.0001 --is_restore=True --model_dir './logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_6_L2Norm_polygon_speed_15_1101202349_False_sigma9_hopk4_adam_tanh__apool_0.0001/' --model_filename 'models-0.1549-4746' &&
#python mgrnn_train_hist_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'tanh' --pool_type '_apool' --drop_out 0.2 --use_curriculum_learning=True --sigma 9 --hopk 6 --optimizer 'adam' --shuffle_training=True --trace_ratio 0.0001 --is_restore=True --model_dir './logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_6_L2Norm_polygon_speed_15_1101202346_False_sigma9_hopk6_adam_tanh__apool_0.0001/' --model_filename 'models-0.1373-6780' &&
#python mgrnn_train_hist_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'tanh' --pool_type '_apool' --drop_out 0.2 --use_curriculum_learning=True --sigma 9 --hopk 8 --optimizer 'adam' --shuffle_training=True --trace_ratio 0.0001 --is_restore=True --model_dir './logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_6_L2Norm_polygon_speed_15_1101202355_False_sigma9_hopk8_adam_tanh__apool_0.0001/' --model_filename 'models-0.1368-4746' &&
#python mgrnn_train_hist_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'tanh' --pool_type '_apool' --drop_out 0.2 --use_curriculum_learning=True --sigma 9 --hopk 10 --optimizer 'adam' --shuffle_training=True --trace_ratio 0.0001 --is_restore=True --model_dir './logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_6_L2Norm_polygon_speed_15_1101202342_False_sigma9_hopk10_adam_tanh__apool_0.0001/' --model_filename 'models-0.1389-4746' &&
## varying sigma
#python mgrnn_train_hist_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'tanh' --pool_type '_apool' --drop_out 0.2 --use_curriculum_learning=True --sigma 6 --hopk 2 --optimizer 'adam' --shuffle_training=True --trace_ratio 0.0001 --is_restore=True --model_dir './logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_6_L2Norm_polygon_speed_15_1104183455_False_sigma6_hopk2_adam_tanh__apool_0.0001/' --model_filename 'models-0.1299-6102' &&
#python mgrnn_train_hist_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'tanh' --pool_type '_apool' --drop_out 0.2 --use_curriculum_learning=True --sigma 3 --hopk 2 --optimizer 'adam' --shuffle_training=True --trace_ratio 0.0001 --is_restore=True --model_dir './logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_6_L2Norm_polygon_speed_15_1104183455_False_sigma3_hopk2_adam_tanh__apool_0.0001/' --model_filename 'models-0.1572-5876' &&
## srun -n 1 --exclusive python mgrnn_train_hist_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'tanh' --pool_type '_apool' --drop_out 0.2 --use_curriculum_learning=True --sigma 9 --hopk 2 --optimizer 'adam' --shuffle_training=True --trace_ratio 0.0001 --is_restore=True --model_dir './logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_6_L2Norm_polygon_speed_15_1104183455_False_sigma9_hopk2_adam_tanh__apool_0.0001/' --model_filename 'models-0.1490-5424' &
#python mgrnn_train_hist_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'tanh' --pool_type '_apool' --drop_out 0.2 --use_curriculum_learning=True --sigma 12 --hopk 2 --optimizer 'adam' --shuffle_training=True --trace_ratio 0.0001 --is_restore=True --model_dir './logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_6_L2Norm_polygon_speed_15_1104183455_False_sigma12_hopk2_adam_tanh__apool_0.0001/' --model_filename 'models-0.1526-4520' &&
#python mgrnn_train_hist_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'tanh' --pool_type '_apool' --drop_out 0.2 --use_curriculum_learning=True --sigma 15 --hopk 2 --optimizer 'adam' --shuffle_training=True --trace_ratio 0.0001 --is_restore=True --model_dir './logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_6_L2Norm_polygon_speed_15_1104183455_False_sigma15_hopk2_adam_tanh__apool_0.0001/' --model_filename 'models-0.1505-5424' &&
#python mgrnn_train_hist_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'tanh' --pool_type '_apool' --drop_out 0.2 --use_curriculum_learning=True --sigma 18 --hopk 2 --optimizer 'adam' --shuffle_training=True --trace_ratio 0.0001 --is_restore=True --model_dir './logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_6_L2Norm_polygon_speed_15_1104183455_False_sigma18_hopk2_adam_tanh__apool_0.0001/' --model_filename 'models-0.1384-6102'



#wait