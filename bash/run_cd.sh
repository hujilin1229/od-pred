#!/usr/bin/env bash

python mgrnn_train_hist_cd.py --test_every_n_epochs 10 --sample_rate 15 --data_format 'speed' --seq_len 3 --horizon 3 \
--num_gpus 2 --fill_mean=False --sparse_removal=False --learning_rate 0.001 --lr_decay 0.8 --lr_decay_epoch 5 \
--batch_size 8 --loss_func 'L2Norm' --activate_func 'tanh' --pool_type '_apool' --drop_out 0.2 \
--use_curriculum_learning=True --sigma 9 --hopk 2 --optimizer 'adam' --shuffle_training=True --trace_ratio 0.0001 \
--is_restore=True --model_dir './logs/mgrnn_L_h_3_32_8-32_4_lr_0.001_LrDecay0.8_LDE5_bs_8_d_0.2_sl_3_L2Norm_polygon_speed_15_1105212839_False_sigma9_hopk2_adam_tanh__apool_0.0001/' \
--model_filename 'models-0.1360-6412'