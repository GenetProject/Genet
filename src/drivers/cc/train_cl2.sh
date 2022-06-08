#!/bin/bash

set -e
save_dir=../results

exp_name=cl2
total_step=720001
pretrain_model_path=../../results_0826/udr_6/udr_start/seed_20/model_step_21600.ckpt
config_file=../../config/train/udr_7_dims_0826/udr_large.json

for seed in 10 20 30;do
    mpiexec -np 2 python train.py \
        --exp-name ${exp_name} \
        --save-dir ${save_dir}/${exp_name}/seed_${seed} \
        --seed ${seed} \
        --total-timesteps $total_step \
        --validation \
        --pretrained-model-path  ${pretrain_model_path} \
        cl2 \
        --baseline cubic \
        --config-file ${config_file}
done
