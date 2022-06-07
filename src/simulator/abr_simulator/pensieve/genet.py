import argparse
import copy
import json
import subprocess
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from typing import Callable, Dict, List, Set, Union

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.observer import _Tracker
from bayes_opt.event import Events

from simulator.abr_simulator.abr_trace import generate_trace
from simulator.abr_simulator.utils import map_log_to_lin, latest_actor_from, map_to_unnormalize
from common.utils import (
    read_json_file, set_seed, write_json_file)
from simulator.abr_simulator.mpc import RobustMPC
from simulator.abr_simulator.bba import BBA
from simulator.abr_simulator.pensieve.pensieve import Pensieve


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("BO training in simulator.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="directory to save testing and intermediate results.")
    parser.add_argument('--model-path', type=str, default="",
                        help="path to Aurora model to start from.")
    parser.add_argument('--video-size-file-dir', type=str,
                        help="path to Aurora model to start from.")
    parser.add_argument("--config-file", type=str,
                        help="Path to configuration file.")
    parser.add_argument("--bo-rounds", type=int, default=20,
                        help="Rounds of BO.")
    parser.add_argument("--nagent", type=int, default=10,
                        help="Number of agenets used in training.")
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--heuristic', type=str, default="mpc",
                        choices=('mpc', 'bba'), help='ABR rule based method.')
    parser.add_argument("--jump-action", action="store_true",
                        help="Use jump action when specified.")
    parser.add_argument(
        "--val-trace-dir",
        type=str,
        default="",
        help="A directory contains the validation trace files.",
    )

    return parser.parse_args()


class JSONLogger(_Tracker):
    def __init__(self, path):
        self._path = path if path[-5:] == ".json" else path + ".json"
        try:
            os.remove(self._path)
        except OSError:
            pass
        super(JSONLogger, self).__init__()

    def update(self, event, instance):
        if event == Events.OPTIMIZATION_STEP:
            data = dict(instance.res[-1])

            now, time_elapsed, time_delta = self._time_metrics()
            data["datetime"] = {
                "datetime": now,
                "elapsed": time_elapsed,
                "delta": time_delta,
            }
            data2dump = copy.deepcopy(data)
            data2dump['params']['max_bw'] = np.exp(data2dump['params']['max_bw'])

            with open(self._path, "a") as f:
                f.write(json.dumps(data2dump) + "\n")

        self._update_tracker(event, instance)


class RandomizationRanges:
    """Manage randomization ranges used in GENET training."""

    def __init__(self, filename: str):
        self.filename = filename
        if filename and os.path.exists(filename):
            self.rand_ranges = read_json_file(filename)
            assert isinstance(self.rand_ranges, List) and len(
                self.rand_ranges) >= 1, "rand_ranges object should be a list with length at least 1."
            weight_sum = 0
            for rand_range in self.rand_ranges:
                weight_sum += rand_range['weight']
            assert weight_sum == 1.0, "Weight sum should be 1."
            self.parameters = set(self.rand_ranges[0].keys())
            self.parameters.remove('weight')
            self.parameters.remove('duration')
        else:
            self.rand_ranges = []
            self.parameters = set()

    def add_ranges(self, range_maps: List[Dict[str, Union[List[float], float]]],
                   prob: float = 0.3) -> None:
        """Add a list of ranges into the randomization ranges.

        The sum of weights of newlly added ranges is prob.
        """
        for rand_range in self.rand_ranges:
            rand_range['weight'] *= (1 - prob)
        if self.rand_ranges:
            weight = prob / len(range_maps)
        else:
            weight = 1 / len(range_maps)
        for range_map in range_maps:
            range_map_to_add = dict()
            for param in self.parameters:

                assert param in range_map, "range_map does not contain '{}'".format(
                    param)
                if param == 'max_bw':
                    range_map_to_add[param] = [np.exp(range_map[param]), np.exp(range_map[param])]
                else:
                    range_map_to_add[param] = [range_map[param], range_map[param]]
            range_map_to_add['weight'] = weight
            range_map_to_add['duration'] = 330
            self.rand_ranges.append(range_map_to_add)

    def get_original_range(self) -> Dict[str, List[float]]:
        start_range = dict()
        for param_name in self.parameters:
            start_range[param_name] = self.rand_ranges[0][param_name]
        return start_range

    def get_parameter_names(self) -> Set[str]:
        return self.parameters

    def get_ranges(self) -> List[Dict[str, List[float]]]:
        return self.rand_ranges

    def dump(self, filename: str) -> None:
        write_json_file(filename, self.rand_ranges)


def black_box_function(min_bw, max_bw, bw_change_interval, link_rtt,
                       buffer_thresh, duration, heuristic, model_path,
                       video_size_file_dir, jump_action, save_dir=""):
    '''
    :param x: input is the current params
    :return: reward is rule-based solution - rl reward
    '''
    max_bw = np.exp(max_bw)
    traces = [generate_trace(bw_change_interval, duration, min_bw, max_bw,
                             link_rtt, buffer_thresh) for _ in range(10)]
    save_dirs = [os.path.join(save_dir, 'trace_{}'.format(i)) for i in range(10)]

    hrewards = heuristic.test_on_traces(traces, video_size_file_dir, save_dirs)
    if jump_action:
        pensieve = Pensieve(model_path, 6, 6, 3)
    else:
        pensieve = Pensieve(model_path)
    rlrewards = pensieve.test_on_traces(traces, video_size_file_dir, save_dirs)

    gap = np.mean(hrewards) - np.mean(rlrewards)
    return gap


class Genet:

    def __init__(self, config_file: str, save_dir: str,
                 black_box_function: Callable, heuristic,
                 model_path: str, video_size_file_dir: str, nagent: int = 10,
                 n_init_pts: int = 10, n_iter: int = 5, seed: int = 42,
                 jump_action: bool = False):

        self.config_file = config_file
        self.cur_config_file = config_file
        self.n_init_pts = n_init_pts
        self.n_iter = n_iter
        self.black_box_function = black_box_function
        self.heuristic = heuristic
        self.seed = seed
        self.nagent = nagent
        self.model_path = model_path
        self.start_model_path = model_path
        self.save_dir = save_dir
        self.video_size_file_dir = video_size_file_dir
        self.jump_action = jump_action

        self.rand_ranges = RandomizationRanges(self.config_file)
        self.param_names = self.rand_ranges.get_parameter_names()
        self.pbounds = copy.deepcopy(self.rand_ranges.get_original_range())
        if 'max_bw' in self.pbounds:
            self.pbounds['max_bw'][0] = np.log(self.pbounds['max_bw'][0])
            self.pbounds['max_bw'][1] = np.log(self.pbounds['max_bw'][1])

    def train(self, rounds: int, epoch_per_round: int, val_dir: str,
              model_save_interval: int = 100):
        """
        """
        # BO guided training flow:
        for i in range(rounds):
            duration = 330
            training_save_dir = os.path.join(self.save_dir, "bo_{}".format(i))
            os.makedirs(training_save_dir, exist_ok=True)
            optimizer = BayesianOptimization(
                    f=lambda min_bw, max_bw, bw_change_interval, link_rtt,
                    buffer_thresh: self.black_box_function(
                    min_bw, max_bw, bw_change_interval, link_rtt, buffer_thresh,
                    duration=duration, heuristic=self.heuristic,
                    model_path=self.model_path, jump_action=self.jump_action,
                    video_size_file_dir=self.video_size_file_dir,
                    save_dir=os.path.join(training_save_dir, 'bo_traces')),
                    pbounds=self.pbounds,
                    random_state=self.seed+i)

            logger = JSONLogger(path=os.path.join(
                self.save_dir, "bo_{}_logs.json".format(i)))
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            optimizer.maximize(
                init_points=self.n_init_pts,
                n_iter=self.n_iter,
                kappa=20,
                xi=0.1
            )
            best_param = optimizer.max
            best_param['params']
            self.rand_ranges.add_ranges([best_param['params']])
            self.cur_config_file = os.path.join(
                self.save_dir, "bo_"+str(i) + ".json")
            self.rand_ranges.dump(self.cur_config_file)

            cmd = "python abr_simulator/pensieve/train.py " \
                    "--total-epoch={total_epoch} " \
                    "--seed={seed} " \
                    "--save-dir={save_dir} " \
                    "--exp-name={exp_name} " \
                    "--model-path={model_path} " \
                    "--nagent={nagent} " \
                    "--video-size-file-dir={video_size_file_dir} "\
                    "--val-freq={val_freq} ".format(
                        total_epoch=epoch_per_round,
                        seed=self.seed,
                        save_dir=training_save_dir,
                        exp_name='bo_{}'.format(i),
                        model_path=self.model_path,
                        nagent=self.nagent,
                        video_size_file_dir=self.video_size_file_dir,
                        val_freq=model_save_interval)

            if self.jump_action:
                cmd += "--jump-action "

            cmd += "udr " \
                    "--config-file={config_file} " \
                    "--val-trace-dir={val_dir}".format(
                        config_file=self.cur_config_file,
                        val_dir=val_dir)
            print(cmd)
            subprocess.run(cmd.split(' '))
            tmp_model_path = latest_actor_from(
                os.path.join(training_save_dir, "model_saved"))
            if tmp_model_path:
                self.model_path = tmp_model_path
            print('current model', self.model_path)


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.heuristic == 'mpc':
        heuristic = RobustMPC()
    elif args.heuristic == 'bba':
        heuristic = BBA()
    else:
        raise NotImplementedError

    genet = Genet(args.config_file, args.save_dir, black_box_function,
                  heuristic, args.model_path, args.video_size_file_dir,
                  nagent=args.nagent, seed=args.seed,
                  jump_action=args.jump_action)

    genet.train(args.bo_rounds, epoch_per_round=5000, val_dir=args.val_trace_dir)


if __name__ == '__main__':
    main()
