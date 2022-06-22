#!/bin/bash


# The architecture of emulation experiment.

#     localhost                |                 mahimahi container(shell)
#                              |
#  HTTP File server <-----video data, bitrate -----> virtual browser (run html, javascript)
#                              |                     ^
#                              |                     |
#                              |           state, bitrate decision
#                              |                     |
#                              |                     V
#                              |               abr(RL, MPC) server

set -e

video_size_dir=data/abr/video_sizes
# ACTOR_PATH=${ROOT}/results/7_dims_rand_large_range_correct_rebuf_penalty/even_udr_1_rand_interval/actor_ep_50000.pth
actor_path=adr_mahimahi/pensieve/data/model_example/ADR_model/nn_model_ep_25600.ckpt
up_link_speed_file=data/abr/12mbps
trace_dir=data/abr/trace_set_1
save_dir=results/abr/emulation


# cd ${ROOT}/pensieve/video_server
# python -m http.server &
# cd ${ROOT}

trap "pkill -f abr_server" SIGINT
trap "pkill -f abr_server" EXIT
# trap "pkill -f abr_server && pkill -f 'python -m http.server'" SIGINT
# trap "pkill -f abr_server && pkill -f 'python -m http.server'" EXIT

delay=40
up_pkt_loss=0
down_pkt_loss=0
buf_th=60
trace_files=`ls ${trace_dir}`
for trace_file in ${trace_files}; do
    for abr in RL RobustMPC BBA; do
        mm-delay ${delay} \
        mm-loss uplink ${up_pkt_loss} \
        mm-loss downlink ${down_pkt_loss} \
        mm-link ${up_link_speed_file} ${trace_dir}/${trace_file} -- \
        bash -c "python src/emulator/abr/virtual_browser/virtual_browser.py \
                        --ip \${MAHIMAHI_BASE} \
                        --port 8000 \
                        --abr ${abr} \
                        --video-size-file-dir ${video_size_dir} \
                        --summary-dir ${save_dir}/${buf_th}_${delay}/${abr} \
                        --trace-file ${trace_file} \
                        --abr-server-port=8322 \
                        --actor-path ${actor_path}"
    done
done
