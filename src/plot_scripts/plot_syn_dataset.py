import csv
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from common.utils import compute_std_of_mean, load_summary, pcc_aurora_reward
from simulator.trace import Trace
from simulator.synthetic_dataset import SyntheticDataset

# SAVE_DIR = '/datamirror/zxxia/PCC-RL/results_1006/evaluate_synthetic_dataset'

# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 42
plt.rcParams['axes.labelsize'] = 42
plt.rcParams['legend.fontsize'] = 42
# plt.rcParams['legend.columnspacing'] = 0.5
# plt.rcParams['legend.labelspacing'] = 0.002
plt.rcParams['figure.figsize'] = (11, 9)

def load_genet_results(save_dirs, bo, seeds, name):
    avg_rewards_across_seeds = []
    avg_tputs_across_seeds = []
    avg_lats_across_seeds = []
    avg_losses_across_seeds = []

    rewards, tputs, lats, losses = [], [], [], []
    reward_errs, tput_errs, lat_errs, loss_errs = [], [], [], []

    for genet_seed in seeds:
        tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = load_summaries_across_traces(
            [os.path.join(save_dir, name, "seed_{}".format(genet_seed),
                          'bo_{}'.format(bo),  'step_64800',
                          'aurora_summary.csv') for save_dir in save_dirs])
        avg_rewards_across_seeds.append(np.mean(np.array(tmp_rewards)))
        avg_tputs_across_seeds.append(np.mean(np.array(tmp_tputs)))
        avg_lats_across_seeds.append(np.mean(np.array(tmp_lats)))
        avg_losses_across_seeds.append(np.mean(np.array(tmp_losses)))
    if name == 'genet_bbr_old_cpu':
        print(avg_rewards_across_seeds)


    rewards.append(np.nanmean(np.array(avg_rewards_across_seeds)))
    reward_errs.append(compute_std_of_mean(avg_rewards_across_seeds))

    tputs.append(np.nanmean(np.array(avg_tputs_across_seeds)))
    tput_errs.append(compute_std_of_mean(avg_tputs_across_seeds))

    lats.append(np.nanmean(np.array(avg_lats_across_seeds)))
    lat_errs.append(compute_std_of_mean(avg_lats_across_seeds))

    losses.append(np.nanmean(np.array(avg_losses_across_seeds)))
    loss_errs.append(compute_std_of_mean(avg_losses_across_seeds))

    low_bnd = np.array(rewards) - np.array(reward_errs)
    up_bnd = np.array(rewards) + np.array(reward_errs)

    tputs_low_bnd = np.array(tputs) - np.array(tput_errs)
    tputs_up_bnd = np.array(tputs) + np.array(tput_errs)

    lats_low_bnd = np.array(lats) - np.array(lat_errs)
    lats_up_bnd = np.array(lats) + np.array(lat_errs)

    losses_low_bnd = np.array(losses) - np.array(loss_errs)
    losses_up_bnd = np.array(losses) + np.array(loss_errs)

    return (rewards, tputs, lats, losses, low_bnd, up_bnd,
            tputs_low_bnd, tputs_up_bnd, lats_low_bnd, lats_up_bnd,
            losses_low_bnd, losses_up_bnd)


def load_summaries_across_traces(log_files: List[str]) -> Tuple[List[float], List[float], List[float], List[float]]:
    rewards = []
    tputs = []
    lats = []
    losses = []
    for log_file in log_files:
        if not os.path.exists(log_file):
            # print(log_file, 'does not exist')
            continue
        summary = load_summary(log_file)
        rewards.append(summary['pkt_level_reward'])
        tputs.append(summary['average_throughput'])
        lats.append(summary['average_latency'] * 1000)
        losses.append(summary['loss_rate'])
        # rewards.append(pcc_aurora_reward(summary['average_throughput'] * 1e6 / 8 / 1500,
        #                summary['average_latency'], summary['loss_rate']))
    return rewards, tputs, lats, losses


def load_results(save_dirs, seeds, steps, name: str):
    rewards, tputs, lats, losses = [], [], [], []
    reward_errs, tput_errs, lat_errs, loss_errs = [], [], [], []
    for step in steps:
        step = int(step)
        avg_rewards_across_seeds = []
        avg_tputs_across_seeds = []
        avg_lats_across_seeds = []
        avg_losses_across_seeds = []

        for seed in seeds:
            if name == 'udr_real':
                if seed == 10:
                    step = 79200
                elif seed == 20:
                    step = 151200
                elif seed == 30:
                    step = 129600
                else:
                    break

            tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = \
                    load_summaries_across_traces([os.path.join(
                save_dir, name, "seed_{}".format(seed), "step_{}".format(step),
                'aurora_summary.csv') for save_dir in save_dirs])
            avg_rewards_across_seeds.append(np.nanmean(np.array(tmp_rewards)))
            avg_tputs_across_seeds.append(np.nanmean(np.array(tmp_tputs)))
            avg_lats_across_seeds.append(np.nanmean(np.array(tmp_lats)))
            avg_losses_across_seeds.append(np.nanmean(np.array(tmp_losses)))

        # print(name, avg_rewards_across_seeds)
        rewards.append(np.nanmean(np.array(avg_rewards_across_seeds)))
        reward_errs.append(compute_std_of_mean(avg_rewards_across_seeds))

        tputs.append(np.nanmean(np.array(avg_tputs_across_seeds)))
        tput_errs.append(compute_std_of_mean(avg_tputs_across_seeds))

        lats.append(np.nanmean(np.array(avg_lats_across_seeds)))
        lat_errs.append(compute_std_of_mean(avg_lats_across_seeds))

        losses.append(np.nanmean(np.array(avg_losses_across_seeds)))
        loss_errs.append(compute_std_of_mean(avg_losses_across_seeds))

    low_bnd = np.array(rewards) - np.array(reward_errs)
    up_bnd = np.array(rewards) + np.array(reward_errs)

    tputs_low_bnd = np.array(tputs) - np.array(tput_errs)
    tputs_up_bnd = np.array(tputs) + np.array(tput_errs)

    lats_low_bnd = np.array(lats) - np.array(lat_errs)
    lats_up_bnd = np.array(lats) + np.array(lat_errs)

    losses_low_bnd = np.array(losses) - np.array(loss_errs)
    losses_up_bnd = np.array(losses) + np.array(loss_errs)
    return (rewards, tputs, lats, losses, low_bnd, up_bnd,
            tputs_low_bnd, tputs_up_bnd, lats_low_bnd, lats_up_bnd,
            losses_low_bnd, losses_up_bnd)


def main():
    save_dirs = []
    dataset = SyntheticDataset.load_from_dir('data/cc/synthetic_dataset')
    save_dirs = [os.path.join('results/cc/evaluate_synthetic_dataset', 'trace_{:05d}'.format(i))
                 for i in range(len(dataset))]

    udr_steps = [720000]
    udr1_rewards, udr1_tputs, udr1_lats, udr1_losses, udr1_low_bnd, udr1_up_bnd, \
    udr1_tputs_low_bnd, udr1_tputs_up_bnd, \
    udr1_lats_low_bnd, udr1_lats_up_bnd, \
    udr1_losses_low_bnd, udr1_losses_up_bnd = load_results(save_dirs, list(range(10, 60, 10)), udr_steps, 'udr1')
    udr1_rewards_err = (udr1_up_bnd[0] - udr1_low_bnd[0]) / 2

    udr2_rewards, udr2_tputs, udr2_lats, udr2_losses, udr2_low_bnd, udr2_up_bnd, \
    udr2_tputs_low_bnd, udr2_tputs_up_bnd, \
    udr2_lats_low_bnd, udr2_lats_up_bnd, \
    udr2_losses_low_bnd, udr2_losses_up_bnd = load_results(save_dirs, [10, 20, 30, 40, 50], udr_steps, 'udr2')
    udr2_rewards_err = (udr2_up_bnd[0] - udr2_low_bnd[0]) / 2

    udr3_rewards, udr3_tputs, udr3_lats, udr3_losses, udr3_low_bnd, udr3_up_bnd, \
    udr3_tputs_low_bnd, udr3_tputs_up_bnd, \
    udr3_lats_low_bnd, udr3_lats_up_bnd, \
    udr3_losses_low_bnd, udr3_losses_up_bnd = load_results(save_dirs, list(range(10, 60, 10)), udr_steps, 'udr3')
    udr3_rewards_err = (udr3_up_bnd[0] - udr3_low_bnd[0]) / 2

    genet_bbr_old_rewards, genet_bbr_old_tputs, genet_bbr_old_lats, genet_bbr_old_losses, genet_bbr_old_low_bnd, genet_bbr_old_up_bnd, \
    genet_bbr_old_tputs_low_bnd, genet_bbr_old_tputs_up_bnd, \
    genet_bbr_old_lats_low_bnd, genet_bbr_old_lats_up_bnd, \
    genet_bbr_old_losses_low_bnd, genet_bbr_old_losses_up_bnd = load_genet_results(save_dirs, 9, [10, 30, 40, 50], 'genet_bbr_old')
    genet_bbr_old_rewards_err = (genet_bbr_old_up_bnd[0] - genet_bbr_old_low_bnd[0]) / 2

    fig = plt.figure()
    ax = plt.gca()
    ax.bar([1, 2, 3], [udr1_rewards[0], udr2_rewards[0],
            udr3_rewards[0]] , yerr=[udr1_rewards_err,
                udr2_rewards_err, udr3_rewards_err])

    ax.bar([4.5], [genet_bbr_old_rewards[0]], yerr=[genet_bbr_old_rewards_err])
    ax.set_xticks([1, 2, 3, 4.5])
    ax.set_xticklabels(['RL1', 'RL2', 'RL3', 'Genet'], rotation=30, ha='right', rotation_mode='anchor')
    ax.set_ylabel('Test reward')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.set_tight_layout(True)
    fig.savefig('fig_reproduce/fig9/fig9_cc.png', bbox_inches='tight')

if __name__ == "__main__":
    main()
