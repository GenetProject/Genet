import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common.utils import compute_std_of_mean
import pandas as pd


plt.style.use('seaborn-deep')
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = 42
# plt.rcParams['axes.labelsize'] = 42
# plt.rcParams['legend.fontsize'] = 42
# plt.rcParams['figure.figsize'] = (11, 9)
plt.rcParams['svg.fonttype'] = 'none'

HATCHES = ['/', '\\', 'x', 'o', '.', 'O', '-', '*', '+']
WIDTH = 0.3



column_wid = 0.7
capsize_wid = 8
eline_wid = 2

def generalization_test_ethernet():
    plt.rcParams['font.size'] = 36
    plt.rcParams['axes.labelsize'] = 36
    plt.rcParams['axes.titlesize'] = 36
    plt.rcParams['legend.fontsize'] = 36
    fig, ax = plt.subplots(figsize=(9, 5))
    df_genet_10 = pd.read_csv('fig_reproduce/fig13/emu_logs/ethernet/queue500/genet/seed_10/summary_no_start_effect.csv')
    df_genet_20 = pd.read_csv('fig_reproduce/fig13/emu_logs/ethernet/queue500/genet/seed_20/summary_no_start_effect.csv')
    df_genet_30 = pd.read_csv('fig_reproduce/fig13/emu_logs/ethernet/queue500/genet/seed_30/summary_no_start_effect.csv')

    genet_reward = np.mean([df_genet_10['aurora_normalized_reward'].mean(), df_genet_20['aurora_normalized_reward'].mean(), df_genet_30['aurora_normalized_reward'].mean()])
    genet_reward_err = np.std([df_genet_10['aurora_normalized_reward'].mean(), df_genet_20['aurora_normalized_reward'].mean(), df_genet_30['aurora_normalized_reward'].mean()]) / np.sqrt(3)

    df_udr1_10 = pd.read_csv('fig_reproduce/fig13/emu_logs/ethernet/queue500/udr1/seed_10/summary_no_start_effect.csv')
    df_udr1_20 = pd.read_csv('fig_reproduce/fig13/emu_logs/ethernet/queue500/udr1/seed_20/summary_no_start_effect.csv')
    df_udr1_30 = pd.read_csv('fig_reproduce/fig13/emu_logs/ethernet/queue500/udr1/seed_30/summary_no_start_effect.csv')

    udr1_reward = np.mean([df_udr1_10['aurora_normalized_reward'].mean(), df_udr1_20['aurora_normalized_reward'].mean(), df_udr1_30['aurora_normalized_reward'].mean()])
    udr1_reward_err = np.std([df_udr1_10['aurora_normalized_reward'].mean(), df_udr1_30['aurora_normalized_reward'].mean(), df_udr1_30['aurora_normalized_reward'].mean()]) / np.sqrt(3)

    df_udr2_10 = pd.read_csv('fig_reproduce/fig13/emu_logs/ethernet/queue500/udr2/seed_10/summary_no_start_effect.csv')
    df_udr2_20 = pd.read_csv('fig_reproduce/fig13/emu_logs/ethernet/queue500/udr2/seed_20/summary_no_start_effect.csv')
    df_udr2_30 = pd.read_csv('fig_reproduce/fig13/emu_logs/ethernet/queue500/udr2/seed_30/summary_no_start_effect.csv')
    
    udr2_reward = np.mean([df_udr2_10['aurora_normalized_reward'].mean(), df_udr2_20['aurora_normalized_reward'].mean(), df_udr2_30['aurora_normalized_reward'].mean()])
    udr2_reward_err = np.std([df_udr2_10['aurora_normalized_reward'].mean(), df_udr2_20['aurora_normalized_reward'].mean(), df_udr2_30['aurora_normalized_reward'].mean()]) / np.sqrt(3)

    df_udr3_10 = pd.read_csv('fig_reproduce/fig13/emu_logs/ethernet/queue500/udr3/seed_10/summary_no_start_effect.csv')
    df_udr3_20 = pd.read_csv('fig_reproduce/fig13/emu_logs/ethernet/queue500/udr3/seed_20/summary_no_start_effect.csv')
    df_udr3_30 = pd.read_csv('fig_reproduce/fig13/emu_logs/ethernet/queue500/udr3/seed_30/summary_no_start_effect.csv')

    udr3_reward = np.mean([df_udr3_10['aurora_normalized_reward'].mean(), df_udr3_20['aurora_normalized_reward'].mean(), df_udr3_30['aurora_normalized_reward'].mean(),])
    udr3_reward_err = np.std([df_udr3_10['aurora_normalized_reward'].mean(), df_udr3_20['aurora_normalized_reward'].mean(), df_udr3_30['aurora_normalized_reward'].mean(),]) / np.sqrt(3)

    df_real_10 = pd.read_csv('fig_reproduce/fig13/emu_logs/ethernet/queue500/real/seed_10/summary_no_start_effect.csv')
    df_real_20 = pd.read_csv('fig_reproduce/fig13/emu_logs/ethernet/queue500/real/seed_20/summary_no_start_effect.csv')
    df_real_30 = pd.read_csv('fig_reproduce/fig13/emu_logs/ethernet/queue500/real/seed_30/summary_no_start_effect.csv')

    real_reward = np.mean([df_real_10['aurora_normalized_reward'].mean(), df_real_20['aurora_normalized_reward'].mean(), df_real_30['aurora_normalized_reward'].mean()])
    real_reward_err = np.std([df_real_10['aurora_normalized_reward'].mean(), df_real_20['aurora_normalized_reward'].mean(), df_real_30['aurora_normalized_reward'].mean()]) / np.sqrt(3)
    
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
