import os

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

from common.utils import load_summary
from simulator.synthetic_dataset import SyntheticDataset

SAVE_DIR  = "results/cc/learning_curve"


def load_summaries_across_traces(log_files: List[str]) -> \
        Tuple[List[float], List[float], List[float], List[float]]:
    rewards = []
    tputs = []
    lats = []
    losses = []
    for log_file in log_files:
        if not os.path.exists(log_file):
            print(log_file, 'does not exist')
            continue
        summary = load_summary(log_file)
        rewards.append(summary['pkt_level_reward'])
        tputs.append(summary['average_throughput'])
        lats.append(summary['average_latency'] * 1000)
        losses.append(summary['loss_rate'])
    return rewards, tputs, lats, losses


def contains_nan_only(l) -> bool:
    for v in l:
        if not np.isnan(v):
            return False
    return True


def load_results(save_dirs: List[str], seeds: List[int], steps: List[int],
                 name: str):
    rewards, tputs, lats, losses = [], [], [], []
    reward_errs, tput_errs, lat_errs, loss_errs = [], [], [], []
    for seed in seeds:
        avg_rewards_across_steps = []
        avg_tputs_across_steps = []
        avg_lats_across_steps = []
        avg_losses_across_steps = []

        for step in steps:
            if name == 'real':
                tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = \
                        load_summaries_across_traces([os.path.join(
                    save_dir, name, "seed_{}".format(seed),
                    'aurora_summary.csv') for save_dir in save_dirs])
            else:
                tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = \
                        load_summaries_across_traces([os.path.join(
                    save_dir, name, "seed_{}".format(seed),
                    "step_{}".format(step), 'aurora_summary.csv')
                    for save_dir in save_dirs])
            avg_rewards_across_steps.append(np.nanmean(np.array(tmp_rewards)))
            avg_tputs_across_steps.append(np.nanmean(np.array(tmp_tputs)))
            avg_lats_across_steps.append(np.nanmean(np.array(tmp_lats)))
            avg_losses_across_steps.append(np.nanmean(np.array(tmp_losses)))

        # avg_rewards_across_steps = list(np.repeat(np.mean(np.array(avg_rewards_across_steps[:48]).reshape(-1, 3), axis=1), 3)) + avg_rewards_across_steps[48:]
        # if name != "pretrained":
        #     avg_rewards_across_steps = avg_rewards_across_steps[:2] + list(np.repeat(np.mean(np.array(avg_rewards_across_steps[2:]).reshape(-1, 3), axis=1), 3))

        rewards.append(np.array(avg_rewards_across_steps))
        tputs.append(np.array(avg_tputs_across_steps))
        lats.append(np.array(avg_lats_across_steps))
        losses.append(np.array(avg_losses_across_steps))

    rewards = np.stack(rewards)
    tputs = np.stack(tputs)
    lats = np.stack(lats)
    losses = np.stack(losses)

    reward_errs = np.std(rewards, axis=0) / np.sqrt(rewards.shape[0])
    tput_errs = np.std(tputs, axis=0) / np.sqrt(tputs.shape[0])
    lat_errs = np.std(lats, axis=0) / np.sqrt(lats.shape[0])
    loss_errs = np.std(losses, axis=0) / np.sqrt(losses.shape[0])
    rewards = np.mean(rewards, axis=0)
    tputs = np.mean(tputs, axis=0)
    lats = np.mean(lats, axis=0)
    losses = np.mean(losses, axis=0)

    low_bnd = np.array(rewards) - np.array(reward_errs)
    up_bnd = np.array(rewards) + np.array(reward_errs)

    tputs_low_bnd = np.array(tputs) - np.array(tput_errs)
    tputs_up_bnd = np.array(tputs) + np.array(tput_errs)

    lats_low_bnd = np.array(lats) - np.array(lat_errs)
    lats_up_bnd = np.array(lats) + np.array(lat_errs)

    losses_low_bnd = np.array(losses) - np.array(loss_errs)
    losses_up_bnd = np.array(losses) + np.array(loss_errs)
    return (list(rewards), list(tputs), list(lats), list(losses), low_bnd, up_bnd,
            tputs_low_bnd, tputs_up_bnd, lats_low_bnd, lats_up_bnd,
            losses_low_bnd, losses_up_bnd)

def load_genet_results(save_dirs: List[str], seeds: List[int], name: str):
    rewards, tputs, lats, losses = [], [], [], []
    reward_errs, tput_errs, lat_errs, loss_errs = [], [], [], []
    for seed in seeds:
        steps = []
        avg_rewards_across_steps = []
        avg_tputs_across_steps = []
        avg_lats_across_steps = []
        avg_losses_across_steps = []
        for bo in range(0, 10):
            # for step in range(0, 72000, 14400):
            for step in range(64800, 72000, 14400):

                tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = \
                    load_summaries_across_traces(
                    [os.path.join(save_dir, name, "seed_{}".format(seed),
                                  'bo_{}'.format(bo), 'step_{}'.format(step),
                                  'aurora_summary.csv') for save_dir in save_dirs])
                avg_rewards_across_steps.append(np.mean(np.array(tmp_rewards)))
                avg_tputs_across_steps.append(np.mean(np.array(tmp_tputs)))
                avg_lats_across_steps.append(np.mean(np.array(tmp_lats)))
                avg_losses_across_steps.append(np.mean(np.array(tmp_losses)))
                steps.append(bo * 72000 + step)

        # avg_rewards_across_steps = avg_rewards_across_steps[:2] + list(np.repeat(np.mean(np.array(avg_rewards_across_steps[2:]).reshape(-1, 3), axis=1), 3))
        rewards.append(np.array(avg_rewards_across_steps))
        tputs.append(np.array(avg_tputs_across_steps))
        lats.append(np.array(avg_lats_across_steps))
        losses.append(np.array(avg_losses_across_steps))

    rewards = np.stack(rewards)
    tputs = np.stack(tputs)
    lats = np.stack(lats)
    losses = np.stack(losses)
    reward_errs = np.std(rewards, axis=0) / np.sqrt(rewards.shape[0])
    tput_errs = np.std(tputs, axis=0) / np.sqrt(tputs.shape[0])
    lat_errs = np.std(lats, axis=0) / np.sqrt(lats.shape[0])
    loss_errs = np.std(losses, axis=0) / np.sqrt(losses.shape[0])
    rewards = np.mean(rewards, axis=0)
    tputs = np.mean(tputs, axis=0)
    lats = np.mean(lats, axis=0)
    losses = np.mean(losses, axis=0)

    low_bnd = np.array(rewards) - np.array(reward_errs)
    up_bnd = np.array(rewards) + np.array(reward_errs)

    tputs_low_bnd = np.array(tputs) - np.array(tput_errs)
    tputs_up_bnd = np.array(tputs) + np.array(tput_errs)

    lats_low_bnd = np.array(lats) - np.array(lat_errs)
    lats_up_bnd = np.array(lats) + np.array(lat_errs)

    losses_low_bnd = np.array(losses) - np.array(loss_errs)
    losses_up_bnd = np.array(losses) + np.array(loss_errs)
    return (steps, list(rewards), list(tputs), list(lats), list(losses), low_bnd, up_bnd,
            tputs_low_bnd, tputs_up_bnd, lats_low_bnd, lats_up_bnd,
            losses_low_bnd, losses_up_bnd)

def main():
    dataset = SyntheticDataset.load_from_dir('data/cc/learning_curve')
    save_dirs = [os.path.join(SAVE_DIR, "trace_{:05d}".format(i)) for i in range(len(dataset))]

    pretrained_steps = [7200, 21600] 
    pretrained_rewards, pretrained_tputs, pretrained_lats, pretrained_losses, \
        pretrained_low_bnd, pretrained_up_bnd, \
    pretrained_tputs_low_bnd, pretrained_tputs_up_bnd, \
    pretrained_lats_low_bnd, pretrained_lats_up_bnd, \
    pretrained_losses_low_bnd, pretrained_losses_up_bnd = load_results(
        save_dirs, list(range(20, 30, 10)), pretrained_steps, 'pretrained')

    genet_bbr_old_steps, genet_bbr_old_rewards, genet_bbr_old_tputs, genet_bbr_old_lats, \
    genet_bbr_old_losses, genet_bbr_old_low_bnd, genet_bbr_old_up_bnd, \
    genet_bbr_old_tputs_low_bnd, genet_bbr_old_tputs_up_bnd, \
    genet_bbr_old_lats_low_bnd, genet_bbr_old_lats_up_bnd, genet_bbr_old_losses_low_bnd, \
    genet_bbr_old_losses_up_bnd = load_genet_results(
            save_dirs, [10, 20, 30], 'genet_bbr_old')

    genet_bbr_old_steps = np.concatenate([np.array(pretrained_steps), pretrained_steps[-1] + np.array(genet_bbr_old_steps)])
    genet_bbr_old_rewards = np.concatenate([pretrained_rewards, genet_bbr_old_rewards])
    genet_bbr_old_low_bnd = np.concatenate([pretrained_low_bnd, genet_bbr_old_low_bnd])
    genet_bbr_old_up_bnd = np.concatenate([pretrained_up_bnd, genet_bbr_old_up_bnd])

    cl1_steps = list(range(0, 720000, 72000))
    cl1_rewards, cl1_tputs, cl1_lats, cl1_losses, \
        cl1_low_bnd, cl1_up_bnd, \
    cl1_tputs_low_bnd, cl1_tputs_up_bnd, \
    cl1_lats_low_bnd, cl1_lats_up_bnd, \
    cl1_losses_low_bnd, cl1_losses_up_bnd = load_results(
        save_dirs, list(range(10, 40, 10)), cl1_steps, 'cl1')
    cl1_steps = np.concatenate([np.array(pretrained_steps), pretrained_steps[-1] + np.array(cl1_steps)])
    cl1_rewards = np.concatenate([pretrained_rewards, cl1_rewards])
    cl1_low_bnd = np.concatenate([pretrained_low_bnd, cl1_low_bnd])
    cl1_up_bnd = np.concatenate([pretrained_up_bnd, cl1_up_bnd])

    cl2_steps = list(range(0, 720000, 72000))
    cl2_rewards, cl2_tputs, cl2_lats, cl2_losses, \
        cl2_low_bnd, cl2_up_bnd, \
    cl2_tputs_low_bnd, cl2_tputs_up_bnd, \
    cl2_lats_low_bnd, cl2_lats_up_bnd, \
    cl2_losses_low_bnd, cl2_losses_up_bnd = load_results(
        save_dirs, list(range(10, 40, 10)), cl2_steps, 'cl2')
    cl2_steps = np.concatenate([np.array(pretrained_steps), pretrained_steps[-1] + np.array(cl2_steps)])
    cl2_rewards = np.concatenate([pretrained_rewards, cl2_rewards])
    cl2_low_bnd = np.concatenate([pretrained_low_bnd, cl2_low_bnd])
    cl2_up_bnd = np.concatenate([pretrained_up_bnd, cl2_up_bnd])

    udr3_steps = list(range(0, 720000, 72000))
    udr3_rewards, udr3_tputs, udr3_lats, udr3_losses, \
        udr3_low_bnd, udr3_up_bnd, \
    udr3_tputs_low_bnd, udr3_tputs_up_bnd, \
    udr3_lats_low_bnd, udr3_lats_up_bnd, \
    udr3_losses_low_bnd, udr3_losses_up_bnd = load_results(
        save_dirs, list(range(10, 40, 10)), udr3_steps, 'udr3')


    cl3_steps, cl3_rewards, cl3_tputs, cl3_lats, \
    cl3_losses, cl3_low_bnd, cl3_up_bnd, \
    cl3_tputs_low_bnd, cl3_tputs_up_bnd, \
    cl3_lats_low_bnd, cl3_lats_up_bnd, cl3_losses_low_bnd, \
    cl3_losses_up_bnd = load_genet_results(
            save_dirs, [10, 20, 30], 'cl3')

    cl3_steps = np.concatenate([np.array(pretrained_steps), pretrained_steps[-1] + np.array(cl3_steps)])
    cl3_rewards = np.concatenate([pretrained_rewards, cl3_rewards])
    cl3_low_bnd = np.concatenate([pretrained_low_bnd, cl3_low_bnd])
    cl3_up_bnd = np.concatenate([pretrained_up_bnd, cl3_up_bnd])

    fig, ax = plt.subplots(1, 1)  # reward curve
    
    ax.plot(genet_bbr_old_steps, genet_bbr_old_rewards, label='Genet')
    ax.fill_between(genet_bbr_old_steps, genet_bbr_old_low_bnd, genet_bbr_old_up_bnd, color='C0', alpha=0.1)

    ax.plot(cl1_steps, cl1_rewards, label='cl1')
    ax.fill_between(cl1_steps, cl1_low_bnd, cl1_up_bnd, color='C1', alpha=0.1)
    ax.plot(cl2_steps, cl2_rewards, label='cl2')
    ax.fill_between(cl2_steps, cl2_low_bnd, cl2_up_bnd, color='C2', alpha=0.1)
    ax.plot(cl3_steps, cl3_rewards, label='cl3')
    ax.fill_between(cl3_steps, cl3_low_bnd, cl3_up_bnd, color='C3', alpha=0.1)

    ax.plot(udr3_steps, udr3_rewards, label='RL3')
    ax.fill_between(udr3_steps, udr3_low_bnd, udr3_up_bnd, color='C4', alpha=0.1)

    ax.set_xlim(0, np.min([cl1_steps[-1], cl2_steps[-1], cl3_steps[-1]]))

    ax.set_xlabel("Training step")
    ax.set_ylabel("Test reward")
    ax.legend(loc='lower right')
    fig.set_tight_layout(True)
    print(genet_bbr_old_steps)
    os.makedirs('fig_reproduce/fig18', exist_ok=True)
    plt.savefig('fig_reproduce/fig18/fig18_cc.png')


if __name__ == '__main__':
    main()
