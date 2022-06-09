#!/bin/bash

set -e

save_dir=results_1006/abr
video_size_file_dir=data/abr/video_sizes
val_trace_dir=data/abr/val_FCC
train_name=cl1
total_epoch=75000


for seed in 10 20 30; do
    python src/simulator/abr_simulator/pensieve/train.py  \
        --save-dir ${save_dir}/${train_name}/seed_${seed} \
        --exp-name ${train_name} \
        --seed ${seed} \
        --total-epoch ${total_epoch} \
        --video-size-file-dir ${video_size_file_dir} \
        --nagent 10 \
        cl1 \
        --val-trace-dir ${val_trace_dir} \
        --config-files config/abr/cl1/difficulty0.json \
        config/abr/cl1/difficulty1.json \
        config/abr/cl1/difficulty2.json \
        config/abr/cl1/difficulty3.json \
        config/abr/cl1/difficulty4.json
done
