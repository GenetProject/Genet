#! /bin/bash

# This file runs rl_test.py on a specified model.

# immediately exit the bash if an error encountered
set -e

SIMULATOR_DIR="../sim"


# new models
NN_MODELS_UDR_3="../data/all_models/udr_3/nn_model_ep_58000.ckpt"
NN_MODELS_UDR_2="../data/all_models/udr_2/nn_model_ep_52400.ckpt"
NN_MODELS_UDR_1="../data/all_models/udr_1/nn_model_ep_57600.ckpt"
NN_MODELS_ADR="../data/all_models/genet/nn_model_ep_9900.ckpt"

NN_MODELS_real="../data/all_models/udr_real/nn_model_ep_49000.ckpt"



TRACE_PATH="../data/more_real_world_2022_trace/fcc-sim-all-cut-filtered/"
SUMMARY_DIR="../sigcomm_artifact/fcc/"

LOG_STR="non"

echo "Running on FCC dataset..."
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

python ${SIMULATOR_DIR}/rl_test_clean_udr_real.py \
       --test_trace_dir ${TRACE_PATH} \
       --summary_dir ${SUMMARY_DIR}\
       --model_path ${NN_MODELS_real} \
       --log_str ${LOG_STR} \
       --random_seed=1

python ${SIMULATOR_DIR}/rl_test_clean_udr_3.py \
       --test_trace_dir ${TRACE_PATH} \
       --summary_dir ${SUMMARY_DIR}\
       --model_path ${NN_MODELS_UDR_3} \
       --log_str ${LOG_STR} \
       --random_seed=1

python plot_results_fcc.py



TRACE_PATH="../data/more_real_world_2022_trace/norway-sim-all-cut-filtered/"
SUMMARY_DIR="../sigcomm_artifact/norway/"

LOG_STR="non"

echo "Running on Norway dataset"
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

python ${SIMULATOR_DIR}/rl_test_clean_udr_real.py \
       --test_trace_dir ${TRACE_PATH} \
       --summary_dir ${SUMMARY_DIR}\
       --model_path ${NN_MODELS_real} \
       --log_str ${LOG_STR} \
       --random_seed=1

python ${SIMULATOR_DIR}/rl_test_clean_udr_3.py \
       --test_trace_dir ${TRACE_PATH} \
       --summary_dir ${SUMMARY_DIR}\
       --model_path ${NN_MODELS_UDR_3} \
       --log_str ${LOG_STR} \
       --random_seed=1 

python plot_results_norway.py
