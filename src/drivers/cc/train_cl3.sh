#!/bin/bash

set -e
save_dir=results/cc
config_file=config/cc/udr3.json
exp_name=cl3
pretrain_model_path=models/cc/pretrained/pretrained.ckpt

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
