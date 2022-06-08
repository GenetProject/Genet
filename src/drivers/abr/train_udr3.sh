#!/bin/bash

set -e
save_dir=/tank/zxxia/PCC-RL/results_1006/abr/new_trace_gen1
# save_dir=/tank/zxxia/PCC-RL/results_1006/abr/jump_action
video_size_file_dir=/tank/zxxia/clean-genet-abr/data/video_sizes/
val_trace_dir=/tank/zxxia/clean-genet-abr/data/BO_stable_traces/test/val_FCC/
total_epoch=75000

train_name=udr3
for seed in 10 20 30; do
    python abr_simulator/pensieve/train.py  \
        --save-dir ${save_dir}/${train_name}/seed_${seed} \
        --exp-name ${train_name} \
        --seed ${seed} \
        --total-epoch ${total_epoch} \
        --video-size-file-dir ${video_size_file_dir} \
        --nagent 10 \
        udr \
        --config-file ../../config/abr/${train_name}.json \
        --val-trace-dir ${val_trace_dir}
done
