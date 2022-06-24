import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

RESULTS_FOLDER = '../emu_results/norway/'
#RESULTS_FOLDER = '../Oboe_results/synthetic/'
TRACE_FOLDER = '../data/real-world-traces/val_Norway/'
NUM_BINS = 100
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 48
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
K_IN_M = 1000.0
REBUF_P = 10
SMOOTH_P = 1
COLOR_MAP = plt.cm.jet #nipy_spectral, Set1,Paired 
SIM_DP = 'sim_dp'
#SCHEMES = ['BB', 'RB', 'FIXED', 'FESTIVE', 'BOLA', 'RL',  'sim_rl', SIM_DP]
#SCHEMES = ['sim_bb', 'sim_mpc', 'sim_rl_pretrain', 'sim_rl_train_noise001', 'sim_rl_train_noise002', 'sim_rl_train_noise003']
#SCHEMES = ['sim_RobustMPC_non', 'sim_adr_non', 'sim_oboe']
SCHEMES = ['sim_BBA', 'sim_RobustMPC', 'sim_udr_1', 'sim_udr_2', 'sim_udr_3', 'sim_adr']
#SCHEMES = ['sim_rl']


def compute_cdf(data):
    """ Return the cdf of input data.

    Args
        data(list): a list of numbers.

    Return
        sorted_data(list): sorted list of numbers.

    """
    length = len(data)
    sorted_data = sorted(data)
    cdf = [i / length for i, val in enumerate(sorted_data)]
    return sorted_data, cdf


def main():
    time_all = {}
    raw_bit_rate_all = {}
    raw_rebuf_all = {}
    bw_all = {}
    raw_reward_all = {}
    raw_smooth_all = {}
    total_rebuf_time={}
    rebuf_ratio = {}

    for scheme in SCHEMES:
        time_all[scheme] = {}
        raw_reward_all[scheme] = {}
        raw_bit_rate_all[scheme] = {}
        raw_rebuf_all[scheme] = {}
        bw_all[scheme] = {}
        raw_smooth_all[scheme] = {}

        total_rebuf_time[scheme] = {}
        rebuf_ratio[scheme] = {}

    log_files = os.listdir(RESULTS_FOLDER)
    for log_file in log_files:

        time_ms = []
        bit_rate = []
        buff = []
        bw = []
        reward = []
        rebuf = []

        smooth =[]

        #print(log_file)

        with open(RESULTS_FOLDER + log_file, 'r') as f:

            for line in f:
                parse = line.split()
                if len(parse) <= 1:
                    break
                time_ms.append(float(parse[0]))
                bit_rate.append(int(parse[1]))
                buff.append(float(parse[2]))
                bw.append(float(parse[4]) / float(parse[5]) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
                rebuf.append(float(parse[3]))
                smooth.append(float(parse[6]))
                reward.append(float(parse[7]))
            #print( reward, "--------------------" )

        time_ms = np.array(time_ms)
        time_ms -= time_ms[0]

        # print log_file

        for scheme in SCHEMES:
            if scheme in log_file:
                time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = time_ms

                raw_bit_rate_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
                raw_rebuf_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = rebuf
                bw_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw
                raw_smooth_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = smooth
                raw_reward_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = reward

                # cal rebuf ratio
                total_rebuf_time[scheme][log_file[len('log_' + str(scheme) + '_'):]] = 0

                for i in raw_rebuf_all[scheme][log_file[len('log_' + str(scheme) + '_'):]]:
                    if i > 0.0:
                        total_rebuf_time[scheme][log_file[len('log_' + str(scheme) + '_'):]] += i

                rebuf_ratio[scheme][log_file[len('log_' + str(scheme) + '_'):]] = \
                    (total_rebuf_time[scheme][log_file[len( 'log_' + str( scheme ) + '_' ):]] /  time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]][-1])

                break

        #print(rebuf_ratio)
    # ---- ---- ---- ----
    # Reward records
    # ---- ---- ---- ----

    log_file_all = []
    reward_all = {}
    bit_rate_all = {}
    rebuf_all = {}
    smooth_all = {}
    rebuf_ratio_all = {}

    for scheme in SCHEMES:
        reward_all[scheme] = []
        bit_rate_all[scheme] = []
        rebuf_all[scheme] = []
        smooth_all[scheme] = []
        rebuf_ratio_all[scheme] = []

    for l in time_all[SCHEMES[0]]:
        # what is l here?
        # l will be something like "norway_ferry_7", representing the name of a trace
        # print(l)

        # assume that the schemes are okay, then flip the flag if they are not
        schemes_check = True

        # all schemes must pass the check
        for scheme in SCHEMES:
            # print(l not in time_all[scheme])
            # check 1: l is a trace name. is the trace name found in every scheme? if not, we fail
            # check 2: is the length of the log for trace "l" less than the video length? if not, we fail
            if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
                # print all the bad ls
                # print(l)
                # print(scheme)
                schemes_check = False
                break
        if schemes_check:
            log_file_all.append(l)
            for scheme in SCHEMES:
                #print(raw_reward_all[scheme], "----------------------")
                reward_all[scheme].append(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN])/VIDEO_LEN)
                bit_rate_all[scheme].append(np.sum(raw_bit_rate_all[scheme][l][1:VIDEO_LEN])/VIDEO_LEN)
                rebuf_all[scheme].append(np.sum(raw_rebuf_all[scheme][l][1:VIDEO_LEN])/VIDEO_LEN)
                smooth_all[scheme].append(np.sum(raw_smooth_all[scheme][l][1:VIDEO_LEN])/VIDEO_LEN)
                # print()
                rebuf_ratio_all[scheme].append((rebuf_ratio[scheme][l]))



    #print(reward_all[scheme], scheme)


    mean_rewards = {}
    error_bar = {}

    mean_bitrate = {}
    mean_rebuf={}
    mean_smooth ={}
    per_rebuf={}
    mean_rebuf_ratio={}

    for scheme in SCHEMES:
        mean_rewards[scheme] = np.mean(reward_all[scheme])
        mean_rewards[scheme] = round(mean_rewards[scheme], 3)
        error_bar[scheme] = np.var(reward_all[scheme])/100
        error_bar[scheme] = round(error_bar[scheme], 4)

        mean_bitrate[scheme] = round(np.mean(bit_rate_all[scheme])/K_IN_M, 2)
        mean_rebuf[scheme] = round(np.mean(rebuf_all[scheme]), 3)
        per_rebuf[scheme] = round(np.percentile(rebuf_all[scheme], 90), 3)
        mean_smooth[scheme] = round(np.mean(smooth_all[scheme]), 3)

        #mean_rebuf_ratio[scheme] = round(np.mean(rebuf_ratio_all[scheme]), 4)
        mean_rebuf_ratio [scheme] = round(np.percentile(rebuf_ratio_all[scheme], 90), 5)

    print(mean_rebuf_ratio, "--------mean_rebuf_ratio")

    fig = plt.figure()
    ax = fig.add_subplot(111)


    for scheme in SCHEMES:
        ax.plot(reward_all[scheme])

    SCHEMES_REW = []
    for scheme in SCHEMES:
        SCHEMES_REW.append(scheme + ': ' + 'bitrate: ' + str(mean_bitrate[scheme])
                           + '% ' + 'rebuf: ' + str(mean_rebuf_ratio[scheme]))

        # SCHEMES_REW.append(scheme + ': ' + str(mean_rewards[scheme]))


    colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    for i,j in enumerate(ax.lines):
        j.set_color(colors[i])

    ax.legend(SCHEMES_REW)
    print(SCHEMES_REW)
    plt.ylabel('Mean reward')
    plt.xlabel('trace index')
    plt.title('Emulation: each dim')
    plt.show()



if __name__ == '__main__':
    main()
