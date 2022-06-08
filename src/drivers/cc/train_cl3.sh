#!/bin/bash

set -e
save_dir=results/cc
config_file=config/train/udr_7_dims_0826/udr_large.json
exp_name=cl3
pretrain_model_path=../../results_0826/udr_6/udr_start/seed_20/model_step_21600.ckpt

for seed in 10 20 30; do
    save_dir=${save_dir}/${exp_name}/seed_${seed}

    python src/simulator/genet_improved.py \
        --heuristic optimal \
        --save-dir ${save_dir} \
        --bo-rounds 15 \
        --seed ${seed} \
        --validation \
        --model-path $pretrain_model_path \
        --config-file  ${config_file}
done
