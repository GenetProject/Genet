import argparse
import logging
import os


def parse_args():
    '''
    Parse arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Pensieve_Retrain")
    parser.add_argument('--description', type=str, default=None,
                        help='Optional description of the experiment.')

    # bit_rate, buffer_size, next_chunk_size, bw_measurement (throughput and time), chunk_til_video_end
    parser.add_argument('--S_INFO', type=int, default='6',
                        help='State info shape. Defaults to 6.')
    parser.add_argument('--S_LEN', type=int, default='6',
                        help='How many frames in the past to consider. Defaults to 8.')
    parser.add_argument('--A_DIM', type=int, default='3',
                        help='Action dimension. Defaults to 3.')
    parser.add_argument( '--BITRATE_DIM' ,type=int ,default='6' ,
                         help='BITRATE dimension. Defaults to 6.' )
    parser.add_argument('--ACTOR_LR_RATE', type=float, default='0.0001',
                        help='Actor learning rate. Defaults to 0.0001.')
    parser.add_argument('--CRITIC_LR_RATE', type=float, default='0.001',
                        help='Critic learning rate. Defaults to 0.001.')
    parser.add_argument('--NUM_AGENTS', type=int, default='8',
                        help='Num of worker agents. Defaults to 16.')

    parser.add_argument('--TRAIN_SEQ_LEN', type=int,
                        default='100', help='take as a train batch')
    parser.add_argument('--MODEL_SAVE_INTERVAL',
                        type=int, default='100', help='')

    parser.add_argument('--RANDOM_SEED', type=int, default='42', help='')

    parser.add_argument('--RAND_RANGE', type=int, default='1000', help='')

    parser.add_argument('--TOTAL_EPOCH', type=int,
                        default='100000', help='total training epoch')

    parser.add_argument('--CHUNK_TIL_VIDEO_END_CAP',
                        type=float, default='48.0', help='')

    parser.add_argument('--BUFFER_NORM_FACTOR', type=float,
                        default='10.0', help='')
    parser.add_argument("--train_trace_dir", type=str,
                        required=True, help='dir to all train traces.')
    parser.add_argument("--val_trace_dir", type=str,
                        required=True, help='dir to all val traces.')
    parser.add_argument("--summary_dir", type=str,
                        required=True, help='output path.')
    parser.add_argument("--nn_model", type=str, default=None,
                        help='model path')
    parser.add_argument("--noise", type=float, default=0,)
    parser.add_argument("--duration", type=float, default=1.0)

    # args for BO
    parser.add_argument("--CURRENT_PARAM", type=float, default=0,
                        help='the param BO output,decide what to train on for the next run')


    return parser.parse_args()


def log_config(args):
    """
    Writes arguments to log. Assumes args.results_dir exists.
    """
    log_file = os.path.join(args.summary_dir, '.config')
    config_logging = logging.getLogger("config")
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    config_logging.setLevel(logging.INFO)
    config_logging.addHandler(file_handler)
    config_logging.addHandler(stream_handler)
    for arg in vars(args):
        config_logging.info(arg + '\t' + str(getattr(args, arg)))
