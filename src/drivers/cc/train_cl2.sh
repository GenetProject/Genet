#!/bin/bash

set -e
save_dir=results/cc
exp_name=cl2
total_step=720001
pretrain_model_path=models/cc/pretrained/pretrained.ckpt
config_file=config/cc/udr3.json

for seed in 10 20 30;do
    mpiexec -np 2 python src/simulator/train.py \
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
