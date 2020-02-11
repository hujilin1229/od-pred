#! /bin/bash
#
#SBATCH --account aauhpc_fat     # account
#SBATCH --nodes 1 		 # number of nodes
#SBATCH --gres=gpu:2             # Number of GPUs per node
#SBATCH --time 24:00:00          # max time (HH:MM:SS)
#SBATCH --cpus-per-task=4       # run two processes on the same node

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

 
# HopK Varying S6H3
python mgrnn_train_hist.py --test_every_n_epochs 1 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 \
--num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 \
--batch_size 8 --loss_func 'L2Norm' --activate_func 'relu' --pool_type '_mpool' --drop_out 0.2 --use_curriculum_learning=True \
--sigma 12 --hopk 4 --optimizer 'adam' --is_restore=True \
--model_dir='./logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_6_L2Norm_taxi_zone_speed_15_1028203017_False_sigma12_hopk4_adam_relu__mpool_0.0001/' \
--model_filename='models-0.0973-54324' &
# srun -n 1 --exclusive python mgrnn_train_hist.py --test_every_n_epochs 1 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'relu' --pool_type '_mpool' --drop_out 0.2 --use_curriculum_learning=True --sigma 12 --hopk 6 --optimizer 'adam' --is_restore=True --model_dir='./logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_6_L2Norm_taxi_zone_speed_15_1030020454_False_sigma12_hopk6_adam_relu__mpool_0.0' --model_filename='models-0.0984-48791' &
# srun -n 1 --exclusive python mgrnn_train_hist.py --test_every_n_epochs 1 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'relu' --pool_type '_mpool' --drop_out 0.2 --use_curriculum_learning=True --sigma 12 --hopk 2 --optimizer 'adam' --is_restore=True --model_dir='./logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_6_L2Norm_taxi_zone_speed_15_1030020454_False_sigma12_hopk2_adam_relu__mpool_0.0/' --model_filename='models-0.0972-62875' &
# srun -n 1 --exclusive python mgrnn_train_hist.py --test_every_n_epochs 1 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'relu' --pool_type '_mpool' --drop_out 0.2 --use_curriculum_learning=True --sigma 12 --hopk 8 --optimizer 'adam' --is_restore=True --model_dir='./logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_6_L2Norm_taxi_zone_speed_15_1030020450_False_sigma12_hopk8_adam_relu__mpool_0.0' --model_filename='models-0.0951-50803' &
# srun -n 1 --exclusive python mgrnn_train_hist.py --test_every_n_epochs 1 --sample_rate 15 --data_format 'speed' --seq_len 6 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'relu' --pool_type '_mpool' --drop_out 0.2 --use_curriculum_learning=True --sigma 12 --hopk 10 --optimizer 'adam' --is_restore=True --model_dir='./logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_6_L2Norm_taxi_zone_speed_15_1030020454_False_sigma12_hopk10_adam_relu__mpool_0.0/' --model_filename='models-0.1009-49797' &

# S3H3
#python mgrnn_train_hist.py --test_every_n_epochs 1 --sample_rate 15 --data_format 'speed' \
#--seq_len 3 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 \
#--lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'relu' --pool_type '_mpool' --drop_out 0.2 \
#--use_curriculum_learning=True --sigma 6 --hopk 2 --optimizer 'adam' --is_restore=True \
#--model_dir='./logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_3_L2Norm_taxi_zone_speed_15_1105212312_False_sigma6_hopk2_adam_relu__mpool_0.0001/' \
#--model_filename='models-0.1037-29854' &

wait


