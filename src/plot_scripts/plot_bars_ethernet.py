import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common.utils import compute_std_of_mean


plt.style.use('seaborn-deep')
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = 42
# plt.rcParams['axes.labelsize'] = 42
# plt.rcParams['legend.fontsize'] = 42
# plt.rcParams['figure.figsize'] = (11, 9)
plt.rcParams['svg.fonttype'] = 'none'

HATCHES = ['/', '\\', 'x', 'o', '.', 'O', '-', '*', '+']
WIDTH = 0.3


genet_reward = 223.88
genet_reward_err = 8.05
udr1_reward = 136.81
udr1_reward_err = 23.61
udr2_reward = 158.48
udr2_reward_err = 17.71
udr3_reward = 159.34
udr3_reward_err = 22.83
real_reward = 191.61
real_reward_err = 3.88      # 26.39   250.47  0.02



column_wid = 0.7
capsize_wid = 8
eline_wid = 2

def generalization_test_ethernet():
    plt.rcParams['font.size'] = 36
    plt.rcParams['axes.labelsize'] = 36
    plt.rcParams['axes.titlesize'] = 36
    plt.rcParams['legend.fontsize'] = 36
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([1, 2, 3, 4],
           [udr1_reward, udr2_reward, udr3_reward, real_reward],
           yerr=[udr1_reward_err, udr2_reward_err, udr3_reward_err,
                 real_reward_err], color='C0', width=column_wid,
           error_kw=dict( lw=eline_wid, capsize=capsize_wid))
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
    fig.savefig('fig_reproduce/fig13/fig13_cc_ethernet.png',  bbox_inches='tight')

if __name__ == '__main__':
    generalization_test_ethernet()
