

#! /bin/bash

# immediately exit the bash if an error encountered
set -e

# run this script in the project root directory
video_size_file_dir=pensieve/data/video_sizes


if [ $(hostname) = "farewell" ]; then
    range_id=0
    summary_dir='/tank/zxxia/active-domainrand/pensieve_results'
    train_config_file=pensieve/config/rand_payload_portion/randomize_payload_portion_range${range_id}.json
    val_config_file='pensieve/config/default.json'
    val_trace_dir=pensieve/data/val
    randomization_interval=1000
    udr_type=udr
    method_name=${udr_type}_${randomization_interval}_rand_interval
    exp_name=randomize_payload_portion_range${range_id}
    python -m pensieve.train_pensieve \
        --video-size-file-dir ${video_size_file_dir} \
        --train-env-config ${train_config_file} \
        --train-trace-dir ${train_trace_dir} \
        --val-env-config ${val_config_file} \
        --summary-dir ${summary_dir}/${exp_name}/${method_name} \
        --randomization ${udr_type} \
        --val-trace-dir ${val_trace_dir} \
        --randomization-interval ${randomization_interval} \
        --model-save-interval 200 \
        --total-epoch 50000
        # --test-env-config ${CONFIG_FILE} \
        # --test-trace-dir ${TEST_TRACE_DIR} \
elif [ $(hostname) = "silver" ]; then
    summary_dir='pensieve/results'
    train_config_file='pensieve/config/randomize_network_params7.json'
    val_config_file='pensieve/config/default.json'
    val_trace_dir=pensieve/data/synthetic_traces/val_rand_network_params
    randomization_interval=1000
    udr_type=udr
    method_name=${udr_type}_${randomization_interval}_rand_interval
    exp_name='randomize_network_params_range7'
    python -m pensieve.train_pensieve \
        --video-size-file-dir ${video_size_file_dir} \
        --train-env-config ${train_config_file} \
        --val-env-config ${val_config_file} \
        --summary-dir ${summary_dir}/${exp_name}/${method_name} \
        --randomization ${udr_type} \
        --val-trace-dir ${val_trace_dir} \
        --randomization-interval ${randomization_interval} \
        --model-save-interval 200 \
        --total-epoch 50000
        # --train-trace-dir ${TRAIN_TRACE_DIR} \
        # --test-env-config ${CONFIG_FILE} \
        # --test-trace-dir ${TEST_TRACE_DIR} \
    echo "in silver"
elif [ $(hostname) = "loon" ]; then
    echo "in loon"
    # VIDEO_SIZE_FILE_DIR='./pensieve/data/video_size_6_larger'
    # CONFIG_FILE='/data3/zxxia/active-domainrand/pensieve/config/randomize_env_parameters.json'
    # CONFIG_FILE='./pensieve/config/randomize_env_parameters1.json'
    # CONFIG_FILE='./pensieve/config/randomize_env_parameters2.json'
    # CONFIG_FILE='./pensieve/config/randomize_parameters_large.json'
    # config_file=pensieve/config/randomize_max_throughput.json
    config_file=pensieve/config/randomize_max_throughput1.json
    # CONFIG_FILE='/data3/zxxia/active-domainrand/pensieve/config/rand_buff_thresh.json'
    # CONFIG_FILE='/data3/zxxia/active-domainrand/pensieve/config/default.json'

    # TRAIN_TRACE_DIR=./pensieve/data/synthetic_traces/train_buf_cap_smart_train
    val_trace_dir=pensieve/data/synthetic_traces/val_rand_max_throughput
    # TEST_TRACE_DIR=./pensieve/data/synthetic_traces/test_buf_cap_smart_train

    summary_dir='pensieve/results'

    # METHOD_NAME='udr_1500_rand_interval'
    # METHOD_NAME='udr_1_rand_interval'
    # METHOD_NAME='even_udr_1500_rand_interval'
    method_name="udr_1000_rand_interval"
    # EXP_NAME='4_dims_rand_no_overlap'
    # EXP_NAME='7_dims_rand_large_range_correct_rebuf_penalty'
    exp_name='1_dim_rand_max_throughput1'
    # EXP_NAME='1_dim_rand_max_throughput_original_bitrate'
    python -m pensieve.train_pensieve \
        --video-size-file-dir ${video_size_file_dir} \
        --train-env-config ${config_file} \
        --val-env-config ${config_file} \
        --val-trace-dir ${val_trace_dir} \
        --summary-dir ${summary_dir}/${exp_name}/${method_name} \
        --randomization-interval 1000 \
        --randomization udr \
        --total-epoch 50000
elif [ $(hostname) = "linux" ] || [ $(hostname) = "linux1" ] || [ $(hostname) = "linux2" ] || [ $(hostname) = "linux3" ]; then
    echo "Linux"
else
    echo "Do nothing"
fi
