import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common.utils import compute_std_of_mean


plt.style.use('seaborn-deep')
plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = 42
# plt.rcParams['axes.labelsize'] = 42
# plt.rcParams['legend.fontsize'] = 42
# plt.rcParams['figure.figsize'] = (11, 9)
plt.rcParams['svg.fonttype'] = 'none'

HATCHES = ['/', '\\', 'x', 'o', '.', 'O', '-', '*', '+']
WIDTH = 0.3

bbr_tput, bbr_tail_lat, bbr_loss = 32.94, 368.93, 0.03
copa_tput, copa_tail_lat, copa_loss = 25.70, 265.02, 0.01
cubic_tput, cubic_tail_lat, cubic_loss = 33.99, 802.69, 0.02
vivace_tput, vivace_tail_lat, vivace_loss = 21.71, 947.25, 0.13
vivace_latency_tput, vivace_latency_tail_lat, vivace_latency_loss =  21.84, 1010.43, 0.13
vivace_loss_tput, vivace_loss_tail_lat, vivace_loss_loss =  28.89, 1125.94, 0.26


genet_tput, genet_tail_lat, genet_loss = 31.77, 183.75, 0.02
udr1_tput, udr1_tail_lat, udr1_loss = 23.16, 204.23, 0.03
udr2_tput, udr2_tail_lat, udr2_loss = 23.09, 185.58, 0.02
udr3_tput, udr3_tail_lat, udr3_loss = 22.72, 179.06, 0.02


column_wid = 0.7
capsize_wid = 8
eline_wid = 2


def cc_scatter():
    plt.rcParams['font.size'] = 32
    plt.rcParams['axes.labelsize'] = 32
    plt.rcParams['axes.titlesize'] = 32
    plt.rcParams['legend.fontsize'] = 32
    fig, ax = plt.subplots(figsize=(9, 5))
    msize = 200
    ax.scatter([bbr_tail_lat], [bbr_tput], marker='d', s=msize, color='C0',
               label='BBR')
    ax.annotate('BBR', (bbr_tail_lat+70, bbr_tput-2))
    ax.scatter([copa_tail_lat], [copa_tput], marker='>', s=msize, color='C1',
               label='Copa')
    ax.annotate('Copa', (copa_tail_lat+110, copa_tput+0.8))
    ax.scatter([cubic_tail_lat], [cubic_tput], marker='v',
               s=msize, color='darkorange', label='Cubic')
    ax.annotate('Cubic', (cubic_tail_lat+80, cubic_tput - 2))
    ax.scatter([vivace_latency_tail_lat], [vivace_latency_tput], marker='^',
               s=msize, color='C3', label='Vivace')
    ax.annotate('Vivace', (vivace_latency_tail_lat, vivace_latency_tput))
    ax.scatter([udr1_tail_lat], [udr1_tput], marker='<', s=msize, color='C4',
               label='RL1')
    ax.annotate('RL1', (udr1_tail_lat+115, udr1_tput))
    ax.scatter([udr2_tail_lat], [udr2_tput], marker='p', s=msize, color='C5',
               label='RL2')
    ax.annotate('RL2', (udr2_tail_lat+20, udr2_tput+0.5))
    ax.scatter([udr3_tail_lat], [udr3_tput], marker='s', s=msize, color='indigo', label='RL3')
    ax.annotate('RL3', (udr3_tail_lat+120, udr3_tput-1.2))
    ax.scatter([genet_tail_lat], [genet_tput], s=msize, color='C2',
               label='Genet')
    ax.annotate('Genet', (genet_tail_lat+80, genet_tput-1.5))
    ax.set_ylabel('Throughput (Mbps)')
    ax.set_xlabel('90th percentile latency (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_xaxis()
	fig.savefig('fig_reproduce/fig_17/fig17_cc_ethernet.png')

if __name__ == '__main__':
	cc_scatter()

