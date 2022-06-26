import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time

import pandas as pd

import numpy as np

from simulator.aurora import test_on_traces
from simulator.network_simulator.bbr import BBR
from simulator.network_simulator.bbr_old import BBR_old
from simulator.network_simulator.cubic import Cubic
# from simulator.network_simulator.pcc.vivace.vivace_latency import VivaceLatency
from simulator.pantheon_dataset import PantheonDataset
from simulator.synthetic_dataset import SyntheticDataset


TRACE_ROOT = "../../data"

TARGET_CCS = ["bbr", "cubic", "vegas", "indigo", "ledbat", "quic"]


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Aurora Testing in simulator.")
    parser.add_argument('--save-dir', type=str, default="",
                        help="direcotry to testing results.")
    parser.add_argument('--dataset-dir', type=str, default="data/synthetic_dataset",
                        help="direcotry to dataset.")
    # parser.add_argument('--cc', type=str, required=True,
                        # choices=("bbr", 'bbr_old', "cubic", "udr1", "udr2", "udr3",
                        #          "genet_bbr", 'genet_bbr_old', 'genet_cubic',
                        #          'real', 'cl1', 'cl2', 'pretrained', 'cl2_new', 'real_cellular', ),
                        # help='congestion control name')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--nproc', type=int, default=16, help='proc cnt')
    parser.add_argument('--fast', action='store_true', help='fast reproduce')

    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    dataset = SyntheticDataset.load_from_dir(args.dataset_dir)
    if args.fast:
        ntraces = 50
        seeds = range(10, 20, 10)
    else:
        ntraces = len(dataset)
        seeds = range(10, 60, 10)
    save_dirs = [os.path.join(args.save_dir, 'trace_{:05d}'.format(i))
                 for i in range(ntraces)]

    for seed in seeds:
        step = 720000
        model_path = "models/cc/udr1/seed_{}/model_step_{}.ckpt".format(seed, step)
        udr_save_dirs = [os.path.join(
            save_dir, 'udr1', "seed_{}".format(seed),
            "step_{}".format(step)) for save_dir in save_dirs]
        test_on_traces(model_path, dataset.traces, udr_save_dirs, args.nproc, 42, False, False)

    for seed in seeds:
        step = 720000
        model_path = "models/cc/udr2/seed_{}/model_step_{}.ckpt".format(seed, step)
        genet_save_dirs = [os.path.join(
            save_dir, 'udr2', "seed_{}".format(seed),
            "step_{}".format(step)) for save_dir in save_dirs]
        test_on_traces(model_path, dataset.traces, genet_save_dirs, args.nproc, 42, False, False)

    for seed in seeds:
        step = 720000
        model_path = "models/cc/udr3/seed_{}/model_step_{}.ckpt".format(seed, step)
        genet_save_dirs = [os.path.join(
            save_dir, 'udr3', "seed_{}".format(seed),
            "step_{}".format(step)) for save_dir in save_dirs]
        test_on_traces(model_path, dataset.traces, genet_save_dirs, args.nproc, 42, False, False)

    for seed in seeds:
        bo = 9
        step  = 64800
        model_path = "models/cc/genet_bbr_old/seed_{}/bo_{}/model_step_{}.ckpt".format(seed, bo, step)
        genet_save_dirs = [os.path.join(
            save_dir, 'genet_bbr_old', "seed_{}".format(seed), "bo_{}".format(bo),
            "step_{}".format(step)) for save_dir in save_dirs]
        test_on_traces(model_path, dataset.traces, genet_save_dirs, args.nproc, 42, False, True)

if __name__ == "__main__":
    t_start = time.time()
    main()
    print('used {} min'.format((time.time() - t_start) / 60))
