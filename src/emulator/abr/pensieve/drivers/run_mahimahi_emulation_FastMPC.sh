#!/bin/bash
set -e

VIDEO_SIZE_DIR=pensieve/data/video_sizes
# ACTOR_PATH=${ROOT}/results/7_dims_rand_large_range_correct_rebuf_penalty/even_udr_1_rand_interval/actor_ep_50000.pth
ACTOR_PATH=pensieve/data/model_example/ADR_model/nn_model_ep_25600.ckpt
UP_LINK_SPEED_FILE=pensieve/data/12mbps
TRACE_DIR=pensieve/data/trace_set_1/
CONFIG_FILE=pensieve/config/emulation/param_sweep.json

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
#for buf_th in $(jq -r -c '.buffer_threshold.values[]' ${CONFIG_FILE}); do
 #   for delay in $(jq -r -c '.delay.values[]' ${CONFIG_FILE}); do
  #      for up_pkt_loss in $(jq -r -c '.uplink_packet_loss_rate.values[]' ${CONFIG_FILE}); do
   #         for down_pkt_loss in $(jq -r -c '.downlink_packet_loss_rate.values[]' ${CONFIG_FILE}); do
                for trace_file in ${trace_files} ; do
		                # trace_file=${UP_LINK_SPEED_FILE}
                    # echo "${buffer_threshold} ${delay} ${up_pkt_loss} ${down_pkt_loss} ${TRACE_FILE}"
                      mm-delay ${delay} mm-loss uplink ${up_pkt_loss} mm-loss downlink ${down_pkt_loss} \
                      mm-link ${UP_LINK_SPEED_FILE} ${TRACE_DIR}${trace_file} -- \
                      bash -c "python -m pensieve.virtual_browser.virtual_browser --ip \${MAHIMAHI_BASE} --port 8012 --abr FastMPC --video-size-file-dir ${VIDEO_SIZE_DIR} --summary-dir pensieve/tests/FastMPC_${buf_th}_${delay}_${TRACE_DIR} --trace-file ${trace_file} --actor-path ${ACTOR_PATH} --abr-server-port=8322"

#                      mm-delay ${delay} mm-loss uplink ${up_pkt_loss} mm-loss downlink ${down_pkt_loss} \
#                      mm-link ${UP_LINK_SPEED_FILE} ${TRACE_DIR}${trace_file} -- \
#                      bash -c "python -m pensieve.virtual_browser.virtual_browser --ip \${MAHIMAHI_BASE} --port 8000 --abr RobustMPC --video-size-file-dir ${VIDEO_SIZE_DIR} --summary-dir pensieve/tests/RL_${buf_th}_${delay}_${TRACE_DIR} --trace-file ${trace_file} --actor-path ${ACTOR_PATH} --abr-server-port=8322"

  #                  mm-delay ${delay} mm-loss uplink ${up_pkt_loss} mm-loss downlink ${down_pkt_loss} \
   #                     mm-link ${UP_LINK_SPEED_FILE} ${TRACE_DIR}/${trace_file} -- \
                      #  bash -c "python -m pensieve.virtual_browser.virtual_browser \
                       #                 --ip \${MAHIMAHI_BASE} \
                        #                --port 8000 \
                         #               --abr RobustMPC \
                          #              --video-size-file-dir ${VIDEO_SIZE_DIR} \
                           #             --summary-dir pensieve/tests/mpc_${buf_th}_${delay}_${up_pkt_loss}_${down_pkt_loss} \
                            #            --trace-file ${trace_file}"
#                done
 #           done
  #      done
  #  done
done
        # pkill -f "python -m http.server"

        # sleep 2
        # mm-delay ${MM_DELAY} mm-link ${UP_LINK_SPEED_FILE} ${TRACE_DIR}/${TRACE_FILE} -- \
        #     bash -c "python -m pensieve.abr_server --abr RobustMPC \
        #                         --video-size-file-dir ${VIDEO_SIZE_DIR} \
        #                         --summary-dir pensieve/tests/mpc_test \
        #                         --trace-file ${TRACE_FILE} --actor-path ${ACTOR_PATH} &
        #             abr_server_pid=\$! &&
        #             python -m pensieve.virtual_browser --ip \${MAHIMAHI_BASE} --port 8000 --abr RL;
        #             kill \${abr_server_pid} && echo kill\${abr_server_pid}"
#     done
# done
