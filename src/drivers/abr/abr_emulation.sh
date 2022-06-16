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

VIDEO_SIZE_DIR=data/abr/video_sizes
# ACTOR_PATH=${ROOT}/results/7_dims_rand_large_range_correct_rebuf_penalty/even_udr_1_rand_interval/actor_ep_50000.pth
ACTOR_PATH=pensieve/data/model_example/ADR_model/nn_model_ep_25600.ckpt
UP_LINK_SPEED_FILE=pensieve/data/12mbps
TRACE_DIR=pensieve/data/trace_set_1/


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
trace_files=`ls ${TRACE_DIR}`
for trace_file in ${trace_files} ; do
    mm-delay ${delay} \
    mm-loss uplink ${up_pkt_loss} \
    mm-loss downlink ${down_pkt_loss} \
    mm-link ${UP_LINK_SPEED_FILE} ${TRACE_DIR}${trace_file} -- \ 
    bash -c "python -m pensieve.virtual_browser.virtual_browser \
                    --ip \${MAHIMAHI_BASE} \
                    --port 8010 \
                    --abr BBA \
                    --video-size-file-dir ${VIDEO_SIZE_DIR} \
                    --summary-dir pensieve/tests/BBA_${buf_th}_${delay}_${TRACE_DIR} \
                    --trace-file ${trace_file} \
                    --abr-server-port=8322"

    mm-delay ${delay} \
    mm-loss uplink ${up_pkt_loss} \
    mm-loss downlink ${down_pkt_loss} \
    mm-link ${UP_LINK_SPEED_FILE} ${TRACE_DIR}${trace_file} -- \ 
    bash -c "python -m pensieve.virtual_browser.virtual_browser \
                    --ip \${MAHIMAHI_BASE} \
                    --port 8010 \
                    --abr RobustMPC \
                    --video-size-file-dir ${VIDEO_SIZE_DIR} \
                    --summary-dir pensieve/tests/BBA_${buf_th}_${delay}_${TRACE_DIR} \
                    --trace-file ${trace_file} \
                    --abr-server-port=8322"

    mm-delay ${delay} \
    mm-loss uplink ${up_pkt_loss} \
    mm-loss downlink ${down_pkt_loss} \
    mm-link ${UP_LINK_SPEED_FILE} ${TRACE_DIR}${trace_file} -- \ 
    bash -c "python -m pensieve.virtual_browser.virtual_browser \
                    --ip \${MAHIMAHI_BASE} \
                    --port 8010 \
                    --abr RL \
                    --video-size-file-dir ${VIDEO_SIZE_DIR} \
                    --summary-dir pensieve/tests/BBA_${buf_th}_${delay}_${TRACE_DIR} \
                    --trace-file ${trace_file} \
                    --abr-server-port=8322"

done
