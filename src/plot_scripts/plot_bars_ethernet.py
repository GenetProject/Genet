import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common.utils import compute_std_of_mean

SAVE_ROOT = '../../figs_sigcomm22'

plt.style.use('seaborn-deep')
plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = 42
# plt.rcParams['axes.labelsize'] = 42
# plt.rcParams['legend.fontsize'] = 42
# plt.rcParams['figure.figsize'] = (11, 9)
plt.rcParams['svg.fonttype'] = 'none'

HATCHES = ['/', '\\', 'x', 'o', '.', 'O', '-', '*', '+']
WIDTH = 0.3

bbr_reward, bbr_tput, bbr_tail_lat, bbr_loss = 192.81, 32.94, 368.93, 0.03
copa_reward, copa_tput, copa_tail_lat, copa_loss = 183.89, 25.70, 265.02, 0.01
cubic_reward, cubic_tput, cubic_tail_lat, cubic_loss = -19.16, 33.99, 802.69, 0.02
vivace_reward, vivace_tput, vivace_tail_lat, vivace_loss = -547.01, 21.71, 947.25, 0.13
vivace_latency_reward, vivace_latency_tput, vivace_latency_tail_lat, vivace_latency_loss = -548.64, 21.84, 1010.43, 0.13
vivace_loss_reward, vivace_loss_tput, vivace_loss_tail_lat, vivace_loss_loss = -825.15, 28.89, 1125.94, 0.26


genet_reward = 223.88
genet_reward_err = 8.05
genet_tput, genet_tail_lat, genet_loss = 31.77, 183.75, 0.02
udr1_reward = 136.81
udr1_reward_err = 23.61
udr1_tput, udr1_tail_lat, udr1_loss = 23.16, 204.23, 0.03
udr2_reward = 158.48
udr2_reward_err = 17.71
udr2_tput, udr2_tail_lat, udr2_loss = 23.09, 185.58, 0.02
udr3_reward = 159.34
udr3_reward_err = 22.83
udr3_tput, udr3_tail_lat, udr3_loss = 22.72, 179.06, 0.02
real_reward = 191.61
real_reward_err = 3.88      # 26.39   250.47  0.02
cl1_reward = 143.86
cl1_reward_err = 7.64      # 22.53   206.07  0.02
cl2_reward = 177.97
cl2_reward_err = 4.55      # 23.17   204.86  0.01


udr3_real_5percent_ethernet_rewards = [177.2, 209.8, 95.2]
udr3_real_10percent_ethernet_rewards = [139, 175, 173]
udr3_real_20percent_ethernet_rewards = [133, 125, 151]
udr3_real_50percent_ethernet_rewards = [162, 124, 78]


column_wid = 0.7
capsize_wid = 8
eline_wid = 2

def generalization_test_ethernet():
    plt.rcParams['font.size'] = 36
    plt.rcParams['axes.labelsize'] = 36
    plt.rcParams['axes.titlesize'] = 36
    plt.rcParams['legend.fontsize'] = 36
    fig, ax = plt.subplots(figsize=(9, 5))
    # plt.bar([1, 2], [bbr_reward, cubic_reward], hatch=HATCHES[:2])
    bars = ax.bar([1, 2, 3, 4],
           [udr1_reward, udr2_reward, udr3_reward, real_reward],
           yerr=[udr1_reward_err, udr2_reward_err, udr3_reward_err,
                 real_reward_err], color='C0', width=column_wid,
           error_kw=dict( lw=eline_wid, capsize=capsize_wid))
    # bars = ax.bar([1, 2, 3, 4],
    #         [udr1_reward, udr2_reward, udr3_reward, real_reward],
    #         color=None, edgecolor='white')
    for bar, pat in zip(bars, HATCHES):
        bar.set_hatch(pat)

    ax.bar([5], [genet_reward], yerr=[genet_reward_err], capsize=8, width=column_wid,
           color='C2', error_kw=dict( lw=eline_wid, capsize=capsize_wid))
    # plt.title('Ethernet')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['RL1', 'RL2', 'RL3', 'RL-real', 'Genet'], rotation=20)
    ax.set_ylabel('Test reward')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False)         # ticks along the top edge are off
    plt.tight_layout()
    fig.savefig('fig_reproduce/fig18_cc_ethernet.png',  bbox_inches='tight')

if __name__ == '__main__':
    generalization_test_ethernet()
