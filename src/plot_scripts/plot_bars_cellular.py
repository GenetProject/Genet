import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.style.use('seaborn-deep')
# plt.rcParams['font.size'] = 42
# plt.rcParams['axes.labelsize'] = 42
# plt.rcParams['legend.fontsize'] = 42
# plt.rcParams['figure.figsize'] = (11, 9)
plt.rcParams['font.size'] = 36
plt.rcParams['axes.labelsize'] = 36
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['legend.fontsize'] = 36
plt.rcParams['svg.fonttype'] = 'none'

HATCHES = ['/', '\\', 'x', 'o', '.', 'O', '-', '*', '+']
WIDTH = 0.3



genet_reward = 252.28
genet_reward_err = 6.46
udr1_reward = 142.31
udr1_reward_err = 23.78     #
udr2_reward = 187.61
udr2_reward_err = 5.03     #
udr3_reward = 203.96
udr3_reward_err = 4.05     # 4.74    386.01  0.01
real_reward = 171.61
real_reward_err = 3.18      # 5.01    459.23  0.02

column_wid = 0.7
capsize_wid = 8
eline_wid = 2

def cellular_bars():
    plt.figure(figsize=(9,5))
    ax = plt.gca()
    bars = plt.bar([1, 2, 3, 4],
        [udr1_reward, udr2_reward, udr3_reward, real_reward],
        yerr=[udr1_reward_err, udr2_reward_err, udr3_reward_err, real_reward_err],
        color='C0', width=column_wid, error_kw=dict( lw=eline_wid, capsize=capsize_wid))
    for bar, pat in zip(bars, HATCHES):
        bar.set_hatch(pat)

    plt.bar([5], [genet_reward], yerr=[genet_reward_err], color='C2',
            width=column_wid, error_kw=dict( lw=eline_wid, capsize=capsize_wid))
    # plt.title('Ethernet')
    ax = plt.gca()
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
    # plt.tight_layout()
    plt.savefig('fig_reproduce/fig13/fig13_cc_cellular.png',  bbox_inches='tight')



if __name__ == '__main__':
    cellular_bars()
    # cc_scatter()
