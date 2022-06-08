#!/bin/bash

set -e
exp_name=udr3
total_step=720001
config_file=../../config/train/udr_large.json
pretrain_model_path=../../results_0826/udr_6/udr_start/seed_20/model_step_21600.ckpt

for seed in 10 20 30; do
    mpiexec -np 2 python train.py \
        --exp-name ${exp_name} \
        --save-dir ${save_dir}/${exp_name}/udr3/seed_${seed} \
        --seed ${seed} \
        --total-timesteps $total_step \
        --validation \
        udr \
        --config-file ../../config/train/udr_large.json
done
