#!/bin/bash

set -e
save_dir=results/cc
exp_name=cl1
total_step=720001
pretrain_model_path=../../results_0826/udr_6/udr_start/seed_20/model_step_21600.ckpt

for seed in 10 20 30;do
    mpiexec -np 2 python src/simulator/train.py \
        --exp-name ${exp_name} \
        --save-dir ${save_dir}/${exp_name}/seed_${seed} \
        --seed ${seed} \
        --total-timesteps $total_step \
        --validation \
        --pretrained-model-path  $pretrain_model_path \
        cl1 \
        --config-files config/train/cl1/difficulty0.json \
                       config/train/cl1/difficulty1.json \
                       config/train/cl1/difficulty2.json \
                       config/train/cl1/difficulty3.json \
                       config/train/cl1/difficulty4.json
done
