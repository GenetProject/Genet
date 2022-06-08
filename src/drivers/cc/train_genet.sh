#!/bin/bash

set -e
save_dir=results/cc
exp_name=genet
config_file=config/train/udr_large.json
pretrain_model_path=../../results_0826/udr_6/udr_start/seed_20/model_step_21600.ckpt

for cc in bbr_old ; do
    for seed in 10 20 30; do
        save_dir=${save_dir}/genet_${cc}/seed_${seed}
        mpiexec -np 2 python src/simulator/genet_improved.py \
            --seed ${seed} \
            --heuristic ${cc} \
            --save-dir ${save_dir} \
            --config-file  ${config_file} \
            --bo-rounds 15 \
            --model-path $pretrain_model_path \
            --validation
    done
done
