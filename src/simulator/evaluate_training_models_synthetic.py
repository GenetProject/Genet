import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import pandas as pd

import numpy as np

from simulator.network_simulator.pcc.aurora.aurora import test_on_traces
from simulator.synthetic_dataset import SyntheticDataset


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Aurora Testing in simulator.")
    parser.add_argument('--dataset-dir', type=str)
    parser.add_argument('--save-dir', type=str, default="",
                        help="direcotry to testing results.")
    parser.add_argument('--cc', type=str, required=True,
                        choices=("udr1", "udr2", "udr3", 'genet_bbr_old',
                                 'cl1', 'cl2', 'cl3', 'pretrained'),
                        help='congestion control name')
    parser.add_argument('--models-path', type=str, default="",
                        help="path to Aurora models.")
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--nproc', type=int, default=16, help='proc cnt')

    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    dataset = SyntheticDataset.load_from_dir(args.dataset_dir)
    traces = dataset.traces
    save_dirs = [os.path.join(args.save_dir, 'trace_{:05d}'.format(i)) for i in range(len(dataset))]

    if args.cc == 'pretrained':
        udr_seed = ''
        for s in args.models_path.split('/'):
            if 'seed' in s:
                udr_seed = s
        # step = 7200
        # while step <= 151200:
        for step in [7200, 21600]:
            if not os.path.exists(os.path.join(args.models_path, 'model_step_{}.ckpt.meta'.format(step))):
                break
            udr_save_dirs = [os.path.join(
                save_dir, args.cc, udr_seed, "step_{}".format(step)) for save_dir in save_dirs]
            model_path = os.path.join(
                args.models_path, 'model_step_{}.ckpt'.format(step))
            test_on_traces(model_path, traces, udr_save_dirs,
                           args.nproc, 42, False, False)
            # step += 28800
    elif args.cc == 'udr1' or args.cc == 'udr2' or args.cc == 'udr3' or \
            args.cc == 'cl1' or args.cc == 'cl1_new' or args.cc == 'cl2' or \
            args.cc == 'cl2_new' or args.cc == 'real_cellular' or \
            'udr' in args.cc:
        udr_seed = ''
        for s in args.models_path.split('/'):
            if 'seed' in s:
                udr_seed = s
        # step = 7200000
        step = 0
        while step <= 720000: #model_step_183240000.ckpt.meta
            if not os.path.exists(os.path.join(args.models_path, 'model_step_{}.ckpt.meta'.format(step))):
                print(os.path.join(args.models_path, 'model_step_{}.ckpt.meta'.format(step)))
                # break
                continue
            udr_save_dirs = [os.path.join(
                save_dir, args.cc, udr_seed, "step_{}".format(step)) for save_dir in save_dirs]
            model_path = os.path.join(
                args.models_path, 'model_step_{}.ckpt'.format(step))
            test_on_traces(model_path, traces, udr_save_dirs,
                           args.nproc, 42, False, True)
            # step += (7200) * 2
            step += 72000
            print(step)
    # elif args.cc == 'genet_bbr' or args.cc == 'genet_cubic' or 'genet_bbr_old':
    elif 'genet' in args.cc or args.cc == 'cl3': #== 'genet_bbr' or args.cc == 'genet_cubic' or 'genet_bbr_old': genet_seed = ''
        for s in args.models_path.split('/'):
            if 'seed' in s:
                genet_seed = s
        # for bo in range(0, 15, 3):
        for bo in range(0, 10):
            bo_dir = os.path.join(args.models_path, "bo_{}".format(bo))
            for step in range(0, 64801, 64800):
            # step = 64800
                model_path = os.path.join(
                    bo_dir, 'model_step_{}.ckpt'.format(step))
                if not os.path.exists(model_path + '.meta'):
                    print("skip " + model_path + '.meta')
                    continue
                genet_save_dirs = [os.path.join(
                    save_dir, args.cc, genet_seed, "bo_{}".format(bo),
                    "step_{}".format(step)) for save_dir in save_dirs]
                # print(genet_save_dirs)
                test_on_traces(model_path, traces, genet_save_dirs,
                               args.nproc, 42, False, False)
    else:
        raise ValueError


if __name__ == "__main__":
    main()
