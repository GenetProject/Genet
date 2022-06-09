#!/bin/bash

set -e
save_dir=results/cc
exp_name=udr3
total_step=720001
config_file=config/cc/${exp_name}.json

for seed in 10 20 30; do
    mpiexec -np 2 python src/simulator/train.py \
        --exp-name ${exp_name} \
        --save-dir ${save_dir}/${exp_name}/udr3/seed_${seed} \
        --seed ${seed} \
        --total-timesteps $total_step \
        --validation \
        udr \
        --config-file ${config_file}
done
