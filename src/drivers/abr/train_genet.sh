#!/bin/bash

set -e

save_dir=/tank/zxxia/PCC-RL/results_1006/abr/new_trace_gen1
# save_dir=/tank/zxxia/PCC-RL/results_1006/abr/jump_action
video_size_file_dir=/tank/zxxia/clean-genet-abr/data/video_sizes/
val_trace_dir=/tank/zxxia/clean-genet-abr/data/BO_stable_traces/test/val_FCC/
config_file=../../config/abr/udr3.json
pretrained_model=/tank/zxxia/PCC-RL/results_1006/abr/new_trace_gen/udr3/seed_10/model_saved/nn_model_ep_1000.ckpt

for seed in 10 20 30; do
    python abr_simulator/pensieve/genet.py \
        --save-dir ${save_dir}/genet_mpc/seed_${seed} \
        --heuristic mpc \
        --seed ${seed} \
        --video-size-file-dir ${video_size_file_dir} \
        --config-file ${config_file} \
        --model-path  ${pretrained_model} \
        --val-trace-dir ${val_trace_dir}
done
