#! /bin/bash

# This file runs rl_test.py on a specified model.

# immediately exit the bash if an error encountered
set -e

SIMULATOR_DIR="../sim"


# new models
NN_MODELS_UDR_3="../data/all_models/udr_3/nn_model_ep_58000.ckpt"
NN_MODELS_UDR_2="../data/all_models/udr_2/nn_model_ep_52400.ckpt"
NN_MODELS_UDR_1="../data/all_models/udr_1/nn_model_ep_57400.ckpt"
NN_MODELS_ADR="../data/all_models/genet/nn_model_ep_9900.ckpt"


TRACE_PATH="../data/synthetic_test/"
SUMMARY_DIR="../sigcomm_artifact/synthetic/"

LOG_STR="non"


python ${SIMULATOR_DIR}/rl_test_clean.py \
       --test_trace_dir ${TRACE_PATH} \
       --summary_dir ${SUMMARY_DIR}\
       --model_path ${NN_MODELS_ADR} \
       --log_str ${LOG_STR} \
       --random_seed=1


python ${SIMULATOR_DIR}/rl_test_clean_udr_1.py \
       --test_trace_dir ${TRACE_PATH} \
       --summary_dir ${SUMMARY_DIR}\
       --model_path ${NN_MODELS_UDR_1} \
       --log_str ${LOG_STR} \
       --random_seed=1

python ${SIMULATOR_DIR}/rl_test_clean_udr_2.py \
       --test_trace_dir ${TRACE_PATH} \
       --summary_dir ${SUMMARY_DIR}\
       --model_path ${NN_MODELS_UDR_2} \
       --log_str ${LOG_STR} \
       --random_seed=1

python ${SIMULATOR_DIR}/rl_test_clean_udr_3.py \
       --test_trace_dir ${TRACE_PATH} \
       --summary_dir ${SUMMARY_DIR}\
       --model_path ${NN_MODELS_UDR_3} \
       --log_str ${LOG_STR} \
       --random_seed=1
python plot_results.py

