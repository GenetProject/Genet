import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

RESULTS_FOLDER = '../sigcomm_artifact/synthetic/'
TRACE_FOLDER = '../data/mahimahi_trace/fcc_mahimahi/'
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
SCHEMES = ['sim_udr_1', 'sim_udr_2', 'sim_udr_3', 'sim_adr']


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
    bit_rate_all = {}
    buff_all = {}
    bw_all = {}
    raw_reward_all = {}

    for scheme in SCHEMES:
        time_all[scheme] = {}
        raw_reward_all[scheme] = {}
        bit_rate_all[scheme] = {}
        buff_all[scheme] = {}
        bw_all[scheme] = {}

    log_files = os.listdir(RESULTS_FOLDER)
    for log_file in log_files:

        time_ms = []
        bit_rate = []
        buff = []
        bw = []
        reward = []

        #print(log_file)

        with open(RESULTS_FOLDER + log_file, 'r') as f:
            if SIM_DP in log_file:
                last_t = 0
                last_b = 0
                last_q = 1
                lines = []
                for line in f:
                    lines.append(line)
                    parse = line.split()
                    if len(parse) >= 6:
                        time_ms.append(float(parse[3]))
                        bit_rate.append(VIDEO_BIT_RATE[int(parse[6])])
                        buff.append(float(parse[4]))
                        bw.append(float(parse[5]))

                for line in reversed(lines):
                    parse = line.split()
                    r = 0
                    if len(parse) > 1:
                        t = float(parse[3])
                        b = float(parse[4])
                        q = int(parse[6])
                        if b == 4:
                            rebuff = (t - last_t) - last_b
                            assert rebuff >= -1e-4
                            r -= REBUF_P * rebuff

                        r += VIDEO_BIT_RATE[q] / K_IN_M
                        r -= SMOOTH_P * np.abs(VIDEO_BIT_RATE[q] - VIDEO_BIT_RATE[last_q]) / K_IN_M
                        reward.append(r)

                        last_t = t
                        last_b = b
                        last_q = q

            else:
                for line in f:
                    parse = line.split()
                    if len(parse) <= 1:
                        break
                    time_ms.append(float(parse[0]))
                    bit_rate.append(int(parse[1]))
                    buff.append(float(parse[2]))
                    bw.append(float(parse[4]) / float(parse[5]) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
                    reward.append(float(parse[-1]))
                #print( reward, "--------------------" )


        if SIM_DP in log_file:
            time_ms = time_ms[::-1]
            bit_rate = bit_rate[::-1]
            buff = buff[::-1]
            bw = bw[::-1]

        time_ms = np.array(time_ms)
        time_ms -= time_ms[0]

        # print log_file

        for scheme in SCHEMES:
            if scheme in log_file:
                time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
                bit_rate_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
                buff_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = buff
                bw_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw
                raw_reward_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = reward
                break

    # ---- ---- ---- ----
    # Reward records
    # ---- ---- ---- ----

    log_file_all = []
    reward_all = {}
    for scheme in SCHEMES:
        reward_all[scheme] = []


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
    #print(reward_all[scheme], scheme)


    mean_rewards = {}
    error_bar = {}
    for scheme in SCHEMES:
        mean_rewards[scheme] = np.mean(reward_all[scheme])
        mean_rewards[scheme] = round(mean_rewards[scheme], 3)
        error_bar[scheme] = np.var(reward_all[scheme])/10
        error_bar[scheme] = round(error_bar[scheme], 4)

    SCHEMES_REW = []
    for scheme in SCHEMES:
        SCHEMES_REW.append(scheme + ': ' + str(mean_rewards[scheme])  + '% ' + str(error_bar[scheme]))
        # SCHEMES_REW.append(scheme + ': ' + str(mean_rewards[scheme]))


    print(SCHEMES_REW)

    #fig ,((ax3 ,ax1 ,ax2)) = plt.subplots( nrows=1 ,ncols=3 )
    fig ,ax1 = plt.subplots()
    column_wid = 0.6
    capsize_wid = 6
    eline_wid = 2.5

    x = [5, 6, 7, 8]

    labels = ['RL1', 'RL2', 'RL3', 'Genet']

    ####### sync

    # ['sim_udr_1: 1.494% 0.0049', 'sim_udr_2: 1.625% 0.0034', 'sim_udr_3: 1.692% 0.0402', 'sim_adr: 2.055% 0.0102']


    UDR_1 = mean_rewards['sim_udr_1'] # [1.50]
    UDR_2 = mean_rewards['sim_udr_2'] # [1.62]
    UDR_3 = mean_rewards['sim_udr_3'] # [1.69]
    genet = mean_rewards['sim_adr'] # [2.05]


    UDR_1_err = error_bar['sim_udr_1'] # [0.005]
    UDR_2_err = error_bar['sim_udr_2'] # [0.003]
    UDR_3_err = error_bar['sim_udr_3'] # [0.04]

    genet_err = error_bar['sim_adr'] # [0.01]



    # ax1.bar( x[0] ,BBA ,yerr=BBA_err ,width=column_wid ,error_kw=dict(lw=eline_wid, capsize=capsize_wid), color='C0' )
    # ax1.bar( x[1] ,MPC ,yerr=MPC_err ,width=column_wid, error_kw=dict(lw=eline_wid, capsize=capsize_wid) , color='C0', hatch='x' )

    ax1.bar( x[0] ,UDR_1 ,yerr=UDR_1_err ,width=column_wid ,error_kw=dict(lw=eline_wid, capsize=capsize_wid),  color='C0' )
    ax1.bar( x[1] ,UDR_2 ,yerr=UDR_2_err ,width=column_wid ,error_kw=dict(lw=eline_wid, capsize=capsize_wid),  color='C0' ,alpha=1, hatch='x'  )
    ax1.bar( x[2] ,UDR_3 ,yerr=UDR_3_err ,width=column_wid ,error_kw=dict(lw=eline_wid, capsize=capsize_wid),  color='C0' ,alpha=1, hatch='/' )

    ax1.bar( x[3] ,genet ,yerr=genet_err ,width=column_wid ,error_kw=dict(lw=eline_wid, capsize=capsize_wid),  color='C2' ,alpha=1 )

    ax1.set_xticks( x )
    ax1.set_xticklabels( labels)

    ax1.set_title( "ABR" )


    labels = ['RL1' ,'RL2' ,'RL3' ,'Genet']

    #
    # bottom_value = -7
    #
    # LLF = [-6.49 + (-bottom_value)]
    # C3 = [-3.74 + (-bottom_value)]
    # UDR_1 = [-4.76 + (-bottom_value)]
    # UDR_2 = [-4.02 + (-bottom_value)]
    # UDR_3 = [-3.69 + (-bottom_value)]
    #
    # genet = [-3.15 + (-bottom_value)]
    #
    # LLF_err = [0.06]
    # C3_err = [0.11]
    #
    # UDR_1_err = [0.24]
    # UDR_2_err = [0.07]
    # UDR_3_err = [0.17]
    #
    # genet_err = [0.12]
    #
    # # ax2.bar( x[0] ,LLF ,yerr=LLF_err ,width=column_wid ,error_kw=dict(lw=eline_wid, capsize=capsize_wid), color='C0' ,bottom=bottom_value )
    # # ax2.bar( x[1] ,C3 ,yerr=C3_err ,width=column_wid ,error_kw=dict(lw=eline_wid, capsize=capsize_wid), color='C0' ,bottom=bottom_value, hatch='x' )
    #
    # ax2.bar( x[0] ,UDR_1 ,yerr=UDR_1_err ,width=column_wid ,error_kw=dict(lw=eline_wid, capsize=capsize_wid), color='C0' ,bottom=bottom_value )
    # ax2.bar( x[1] ,UDR_2 ,yerr=UDR_2_err ,width=column_wid ,error_kw=dict(lw=eline_wid, capsize=capsize_wid), color='C0' ,alpha=1 ,bottom=bottom_value, hatch='x')
    # ax2.bar( x[2] ,UDR_3 ,yerr=UDR_3_err ,width=column_wid ,error_kw=dict(lw=eline_wid, capsize=capsize_wid), color='C0' ,alpha=1 ,bottom=bottom_value, hatch='/' )
    #
    # ax2.bar( x[3] ,genet ,yerr=genet_err ,width=column_wid ,error_kw=dict(lw=eline_wid, capsize=capsize_wid), color='C2' ,alpha=1 ,bottom=bottom_value )
    #
    # ax2.set_xticks( x )
    # ax2.set_title( "LB" )
    # ax2.set_xticklabels( labels)
    #
    # labels = ['RL1' ,'RL2' ,'RL3' ,'Genet']
    #
    #
    # bbr_reward = [69.67]
    # cubic_reward = [-41.84]
    #
    # small_reward = [53.16]
    # small_reward_err = [7.98]
    # medium_reward = [67.85]
    # medium_reward_err = [4.32]
    # large_reward = [65.9]
    # large_reward_err = [2.97]
    #
    # genet_reward = [76.20]
    # genet_reward_err = [3.44]
    #
    # # ax3.bar( x[0] ,bbr_reward ,width=column_wid ,color='C0' )
    # # ax3.bar( x[1] ,cubic_reward,width=column_wid ,color='C0', hatch='x')
    #
    # ax3.bar( x[0] ,small_reward ,yerr=small_reward_err ,error_kw=dict(lw=eline_wid, capsize=capsize_wid), width=column_wid ,color='C0' )
    # ax3.bar( x[1] ,medium_reward ,yerr=medium_reward_err,error_kw=dict(lw=eline_wid, capsize=capsize_wid), width=column_wid ,color='C0' ,alpha=1, hatch='x'  )
    # ax3.bar( x[2] ,large_reward ,yerr=large_reward_err ,error_kw=dict(lw=eline_wid, capsize=capsize_wid), width=column_wid ,color='C0' ,alpha=1, hatch='/'  )
    # ax3.bar( x[3] ,genet_reward ,yerr=genet_reward_err ,error_kw=dict(lw=eline_wid, capsize=capsize_wid), width=column_wid ,color='C2' ,alpha=1)
    #
    # ax3.set_xticks( x )
    # ax3.set_title( "CC" )
    # ax3.set_ylabel( 'Test reward' )
    #
    #
    # ax3.set_xticklabels( labels)

    ax1.spines['right'].set_visible( False )
    ax1.spines['top'].set_visible( False )
    # ax2.spines['right'].set_visible( False )
    # ax2.spines['top'].set_visible( False )
    # ax3.spines['right'].set_visible( False )
    # ax3.spines['top'].set_visible( False )

    # svg_file = os.path.join( SAVE_ROOT ,'output_figs/all_on_sim.svg' )
    # pdf_file = os.path.join( SAVE_ROOT ,'output_figs/all_on_sim.pdf' )
    # fig.savefig( svg_file ,bbox_inches='tight' )
    # os.system( "inkscape {} --export-pdf={}".format( svg_file ,pdf_file ) )
    # os.system( "pdfcrop --margins 1 {} {}".format( pdf_file ,pdf_file ) )
    # plt.show()
    plt.savefig('fig9_abr.png')


if __name__ == '__main__':
    main()
