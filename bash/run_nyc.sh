#!/usr/bin/env bash

python mgrnn_train_hist.py --test_every_n_epochs 1 --sample_rate 15 --data_format 'speed' \
--seq_len 3 --horizon 3 --num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 \
--lr_decay_epoch 5 --batch_size 8 --loss_func 'L2Norm' --activate_func 'relu' --pool_type '_mpool' --drop_out 0.2 \
--use_curriculum_learning=True --sigma 6 --hopk 2 --optimizer 'adam' --is_restore=True \
--model_dir='./logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_3_L2Norm_taxi_zone_speed_15_1105212312_False_sigma6_hopk2_adam_relu__mpool_0.0001/' \
--model_filename='models-0.1037-29854' &