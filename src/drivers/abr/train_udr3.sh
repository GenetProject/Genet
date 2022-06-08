#!/bin/bash

set -e
save_dir=results/abr
video_size_file_dir=data/abr/video_sizes
val_trace_dir=data/abr/val_FCC
total_epoch=75000
train_name=udr3
config_file=config/abr/${train_name}.json

for seed in 10 20 30; do
    python src/simulator/abr_simulator/pensieve/train.py  \
        --save-dir ${save_dir}/${train_name}/seed_${seed} \
        --exp-name ${train_name} \
        --seed ${seed} \
        --total-epoch ${total_epoch} \
        --video-size-file-dir ${video_size_file_dir} \
        --nagent 10 \
        udr \
        --config-file ${config_file} \
        --val-trace-dir ${val_trace_dir}
done
