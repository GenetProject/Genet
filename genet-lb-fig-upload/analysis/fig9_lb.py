import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 34
plt.rcParams['axes.labelsize'] = 34
plt.rcParams['legend.fontsize'] = 34
plt.rcParams['axes.titlesize'] = 34
plt.rcParams["figure.figsize"] = (7,5)
plt.rcParams['svg.fonttype'] = 'none'

SAVE_ROOT = "figs/"

def main():
    # [Mean, std]
    # genet=[-3.02, 0.04], udr_1=[-4.80, 0.07], udr_2=[-3.87, 0.08], udr_3=[-3.57, 0.07],

    fig ,ax2 = plt.subplots(1, 1)
    column_wid = 0.6
    capsize_wid = 6
    eline_wid = 2.5


    x = [5, 6, 7, 8]
    labels = ['RL1' ,'RL2' ,'RL3' ,'Genet']


    bottom_value = -7

    UDR_1 = [-4.80 + (-bottom_value)]
    UDR_2 = [-3.87 + (-bottom_value)]
    UDR_3 = [-3.57 + (-bottom_value)]
    genet = [-3.02 + (-bottom_value)]

    UDR_1_err = [0.07]
    UDR_2_err = [0.08]
    UDR_3_err = [0.07]
    genet_err = [0.04]

    ax2.bar( x[0] ,UDR_1 ,yerr=UDR_1_err ,width=column_wid ,error_kw=dict(lw=eline_wid, capsize=capsize_wid), color='C0' ,bottom=bottom_value )
    ax2.bar( x[1] ,UDR_2 ,yerr=UDR_2_err ,width=column_wid ,error_kw=dict(lw=eline_wid, capsize=capsize_wid), color='C0' ,alpha=1 ,bottom=bottom_value, hatch='x')
    ax2.bar( x[2] ,UDR_3 ,yerr=UDR_3_err ,width=column_wid ,error_kw=dict(lw=eline_wid, capsize=capsize_wid), color='C0' ,alpha=1 ,bottom=bottom_value, hatch='/' )

    ax2.bar( x[3] ,genet ,yerr=genet_err ,width=column_wid ,error_kw=dict(lw=eline_wid, capsize=capsize_wid), color='C2' ,alpha=1 ,bottom=bottom_value )

    ax2.set_xticks( x )
    ax2.set_title( "LB" )
    ax2.set_xticklabels( labels)



    ax2.spines['right'].set_visible( False )
    ax2.spines['top'].set_visible( False )
    plt.savefig("../fig_reproduce/fig9/fig9_lb.png")


if __name__ == '__main__':
    main()
