import argparse
import os
import time
import warnings

from simulator.abr_simulator.pensieve.pensieve import Pensieve
from simulator.abr_simulator.schedulers import (
    UDRTrainScheduler,
    CL1TrainScheduler,
    CL2TrainScheduler,
)
from simulator.abr_simulator.utils import load_traces
from simulator.abr_simulator.abr_trace import AbrTrace
from common.utils import set_seed, save_args

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Training code.")
    parser.add_argument(
        "--jump-action", action="store_true",
        help="Use jump action when specified."
    )
    parser.add_argument(
        "--exp-name", type=str, default="", help="Experiment name."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="direcotry to save the model.",
    )
    parser.add_argument("--seed", type=int, default=20, help="seed")
    parser.add_argument(
        "--total-epoch",
        type=int,
        default=100,
        help="Total number of epoch to be trained.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Path to a pretrained Tensorflow checkpoint.",
    )
    parser.add_argument(
        "--video-size-file-dir",
        type=str,
        default="",
        help="Path to video size files.",
    )
    parser.add_argument(
        "--nagent",
        type=int,
        default=2,
        help="Path to a pretrained Tensorflow checkpoint.",
    )
    # parser.add_argument(
    #     "--validation",
    #     action="store_true",
    #     help="specify to enable validation.",
    # )
    parser.add_argument(
        "--val-freq",
        type=int,
        default=1000,
        help="specify to enable validation.",
    )
    subparsers = parser.add_subparsers(dest="curriculum", help="CL parsers.")
    udr_parser = subparsers.add_parser("udr", help="udr")
    udr_parser.add_argument(
        "--real-trace-prob",
        type=float,
        default=0.0,
        help="Probability of picking a real trace in training",
    )
    udr_parser.add_argument(
        "--train-trace-dir",
        type=str,
        default="",
        help="A directory contains the training trace files.",
    )
    udr_parser.add_argument(
        "--val-trace-dir",
        type=str,
        default="",
        help="A directory contains the validation trace files.",
    )
    udr_parser.add_argument(
        "--config-file",
        type=str,
        default="",
        help="A json file which contains a list of randomization ranges with "
        "their probabilites.",
    )
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default="pantheon",
    #     choices=("pantheon", "synthetic"),
    #     help="dataset name",
    # )
    cl1_parser = subparsers.add_parser("cl1", help="cl1")
    cl1_parser.add_argument(
        "--config-files",
        type=str,
        nargs="+",
        help="A list of randomization config files.",
    )
    cl1_parser.add_argument(
        "--val-trace-dir",
        type=str,
        default="",
        help="A directory contains the validation trace files.",
    )
    cl2_parser = subparsers.add_parser("cl2", help="cl2")
    cl2_parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        choices=("mpc", "bba"),
        help="Baseline used to sort environments.",
    )
    cl2_parser.add_argument(
        "--config-file",
        type=str,
        default="",
        help="A json file which contains a list of randomization ranges with "
        "their probabilites.",
    )
    cl2_parser.add_argument(
        "--val-trace-dir",
        type=str,
        default="",
        help="A directory contains the validation trace files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    assert (
        not args.model_path
        or args.model_path.endswith(".ckpt")
    )
    os.makedirs(args.save_dir, exist_ok=True)
    save_args(args, args.save_dir)
    set_seed(args.seed)

    # Initialize model and agent policy
    if args.jump_action:
        pensieve = Pensieve(args.model_path, 6, 6, 3, train_mode=True)
    else:
        pensieve = Pensieve(args.model_path, train_mode=True)
        # args.seed,
        # args.save_dir,
        # int(args.val_freq / nagents),
        # tensorboard_log=args.tensorboard_log,
    # training_traces, validation_traces,
    training_traces = []
    val_traces = []
    if args.curriculum == "udr":
        config_file = args.config_file
        if args.train_trace_dir:
            all_time, all_bw, all_file_names = load_traces(args.train_trace_dir)
            training_traces = [AbrTrace(t, bw, link_rtt=80, buffer_thresh=60, name=name)
                               for t, bw, name in zip(all_time, all_bw, all_file_names)]

        if args.val_trace_dir:
            all_time, all_bw, all_file_names = load_traces(args.val_trace_dir)
            val_traces = [AbrTrace(t, bw, link_rtt=80, buffer_thresh=60, name=name)
                               for t, bw, name in zip(all_time, all_bw, all_file_names)]
        train_scheduler = UDRTrainScheduler(
            config_file,
            training_traces,
            percent=args.real_trace_prob,
        )
    elif args.curriculum == "cl1":
        train_scheduler = CL1TrainScheduler(args.config_files)
        if args.val_trace_dir:
            all_time, all_bw, all_file_names = load_traces(args.val_trace_dir)
            val_traces = [AbrTrace(t, bw, link_rtt=80, buffer_thresh=60, name=name)
                               for t, bw, name in zip(all_time, all_bw, all_file_names)]
    elif args.curriculum == "cl2":
        config_file = args.config_file
        train_scheduler = CL2TrainScheduler(config_file, args.baseline)
    else:
        raise NotImplementedError

    pensieve.train(
        train_scheduler,
        val_traces,
        args.save_dir,
        args.nagent,
        args.total_epoch,
        args.video_size_file_dir,
        args.val_freq)


if __name__ == "__main__":
    t_start = time.time()
    main()
    print("time used: {:.2f}s".format(time.time() - t_start))
