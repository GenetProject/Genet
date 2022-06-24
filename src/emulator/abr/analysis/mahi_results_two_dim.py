import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-deep')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 32
plt.rcParams['axes.labelsize'] = 32
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['legend.fontsize'] = 32
plt.rcParams["figure.figsize"] = (9,5)
plt.rcParams['svg.fonttype'] = 'none'

SAVE_ROOT = "figs/"

def main():
    os.makedirs( SAVE_ROOT ,exist_ok=True )


    # ['sim_BBA: bitrate: 1.2% rebuf: 0.0585' ,'sim_RobustMPC: bitrate: 1.22% rebuf: 0.033' ,
    #  'sim_udr_1: bitrate: 1.2% rebuf: 0.0338' ,'sim_udr_2: bitrate: 1.04% rebuf: 0.0195' ,
    #  'sim_udr_3: bitrate: 1.1% rebuf: 0.0237' ,'sim_adr: bitrate: 1.11% rebuf: 0.0148']

    plt.rcParams['font.size'] = 32
    plt.rcParams['axes.labelsize'] = 32
    plt.rcParams['axes.titlesize'] = 32
    plt.rcParams['legend.fontsize'] = 32
    fig1, ax1 = plt.subplots(figsize=(9, 5))

    fcc_genet_bitrate, fcc_genet_rebuf = 1.17, 0.014

    fcc_udr1_bitrate ,fcc_udr1_rebuf = 1.2 ,0.034
    fcc_udr2_bitrate ,fcc_udr2_rebuf = 1.04 ,0.019
    fcc_udr3_bitrate ,fcc_udr3_rebuf = 1.1 ,0.023

    fcc_bba_bitrate ,fcc_bba_rebuf = 1.2 , 0.058
    fcc_mpc_bitrate ,fcc_mpc_rebuf = 1.22 ,0.033
    fcc_oboe_bitrate ,fcc_oboe_rebuf = 1.16 , 0.021



    msize = 200

    ax1.scatter( [fcc_bba_rebuf],[fcc_bba_bitrate] ,marker='d' ,color='C0' ,s=msize ,label='BBA' )
    ax1.scatter( [fcc_mpc_rebuf] ,[fcc_mpc_bitrate] ,marker='>' ,color='C1' ,s=msize ,label='MPC' )
    ax1.scatter( [fcc_oboe_rebuf] ,[fcc_oboe_bitrate] ,marker='v' ,color='darkorange' ,s=msize ,label='Oboe' )
    ax1.scatter( [fcc_udr1_rebuf] , [fcc_udr1_bitrate] ,marker='^' ,color='C3' ,s=msize ,label='RL1' )
    ax1.scatter( [fcc_udr2_rebuf] ,[fcc_udr2_bitrate] ,marker='<' ,color='C4' ,s=msize ,label='RL2' )
    ax1.scatter( [fcc_udr3_rebuf] , [fcc_udr3_bitrate] , marker='p' ,color='C5' ,s=msize ,label='RL3' )
    ax1.scatter( [fcc_genet_rebuf] ,[fcc_genet_bitrate] ,s=msize ,color='C2' ,label='Genet' )

    ax1.annotate('BBA', ( fcc_bba_rebuf, fcc_bba_bitrate-0.03))
    ax1.annotate('MPC', ( fcc_mpc_rebuf, fcc_mpc_bitrate-0.01))
    ax1.annotate('Oboe', (fcc_oboe_rebuf+0.004, fcc_oboe_bitrate-0.025))
    ax1.annotate('RL1', (fcc_udr1_rebuf+0.002, fcc_udr1_bitrate-0.03))
    ax1.annotate('RL2', (fcc_udr2_rebuf+0.003, fcc_udr2_bitrate+0.01))
    ax1.annotate('RL3', (fcc_udr3_rebuf+0.007, fcc_udr3_bitrate-0.01))
    ax1.annotate('Genet', ( fcc_genet_rebuf+0.005, fcc_genet_bitrate+0.02))

    ax1.set_ylabel('Bitrate (Mbps)')
    ax1.set_xlabel('90th percentile rebuffering ratio (%)')
    ax1.invert_xaxis()
    ax1.spines['top'].set_visible( False )
    ax1.spines['right'].set_visible( False )
    #ax.set_ylim(0.03, -0.01)

    fig1.legend(bbox_to_anchor=(0, 1.02, 1, 0.14), ncol=4, loc="upper center",
                borderaxespad=0, borderpad=0.2, columnspacing=0.01, handletextpad=0.001)

    # svg_file = os.path.join(SAVE_ROOT, 'evaluation_cc_scatter.svg')
    # pdf_file = os.path.join(SAVE_ROOT, 'evaluation_cc_scatter.pdf')
    # fig1.savefig(pdf_file,  bbox_inches='tight')
    # # os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    # # os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))

    svg_file = os.path.join( SAVE_ROOT ,'mahi_fcc_arrow.svg' )
    pdf_file = os.path.join( SAVE_ROOT ,'mahi_fcc_arrow.pdf' )
    fig1.savefig( pdf_file ,bbox_inches='tight' )
    # os.system( "inkscape {} --export-pdf={}".format( svg_file ,pdf_file ) )
    # os.system( "pdfcrop --margins 1 {} {}".format( pdf_file ,pdf_file ) )

    fig2, ax2 = plt.subplots(figsize=(9, 5))

    # ['sim_BBA: bitrate: 1.03% rebuf: 0.07658' ,'sim_RobustMPC: bitrate: 1.05% rebuf: 0.05053' ,
    #  'sim_udr_1: bitrate: 1.04% rebuf: 0.07323' ,'sim_udr_2: bitrate: 0.96% rebuf: 0.04276' ,
    #  'sim_udr_3: bitrate: 0.95% rebuf: 0.04796' ,'sim_adr: bitrate: 0.95% rebuf: 0.04498']

    norway_genet_bitrate ,norway_genet_rebuf = 1.04 ,0.043

    norway_udr1_bitrate ,norway_udr1_rebuf = 1.04 ,0.073
    norway_udr2_bitrate ,norway_udr2_rebuf = 0.96 ,0.043
    norway_udr3_bitrate ,norway_udr3_rebuf = 0.95 ,0.048

    norway_bba_bitrate ,norway_bba_rebuf = 1.03 ,0.077
    norway_mpc_bitrate ,norway_mpc_rebuf = 1.05 ,0.054
    norway_oboe_bitrate ,norway_oboe_rebuf = 1.04 ,0.051

    ax2.scatter( [norway_bba_rebuf], [norway_bba_bitrate] ,marker='d', color='C0', s=msize ,label='BBA' )
    ax2.scatter( [norway_mpc_rebuf] ,[norway_mpc_bitrate] ,marker='>' ,color='C1',s=msize ,label='MPC' )
    ax2.scatter( [norway_oboe_rebuf] ,[norway_oboe_bitrate] ,marker='v' ,color='darkorange',s=msize ,label='Oboe' )
    ax2.scatter( [norway_udr1_rebuf] ,[norway_udr1_bitrate] ,marker='^' ,color='C3',s=msize ,label='RL1' )
    ax2.scatter( [norway_udr2_rebuf] ,[norway_udr2_bitrate] ,marker='<' ,color='C4',s=msize ,label='RL2' )
    ax2.scatter( [norway_udr3_rebuf] ,[norway_udr3_bitrate] ,marker='p' ,color='C5',s=msize ,label='RL3' )
    ax2.scatter( [norway_genet_rebuf] ,[norway_genet_bitrate] , s=msize ,color='C2', label='Genet' )

    ax2.annotate('BBA', (norway_bba_rebuf+0.0001, norway_bba_bitrate-0.015))
    ax2.annotate('MPC', (norway_mpc_rebuf+0.005, norway_mpc_bitrate-0.02))
    ax2.annotate('Oboe', ( norway_oboe_rebuf+0.001, norway_oboe_bitrate-0.015))
    ax2.annotate('RL1', (norway_udr1_rebuf-0.001, norway_udr1_bitrate-0.01))
    ax2.annotate('RL2', (norway_udr2_rebuf+0.003, norway_udr2_bitrate+0.01))
    ax2.annotate('RL3', (norway_udr3_rebuf+0.006, norway_udr3_bitrate))
    ax2.annotate('Genet', (norway_genet_rebuf+0.005, norway_genet_bitrate+0.01))



    ax2.set_ylabel( 'Bitrate (Mbps)' )
    ax2.set_xlabel( '90th percentile rebuffering ratio (%)' )
    ax2.invert_xaxis()
    ax2.spines['top'].set_visible( False )
    ax2.spines['right'].set_visible( False )

    svg_file = os.path.join( SAVE_ROOT ,'mahi_norway_arrow.svg' )
    pdf_file = os.path.join( SAVE_ROOT , 'mahi_norway_arrow.pdf')
    fig2.savefig( pdf_file ,bbox_inches='tight' )
    # os.system( "inkscape {} --export-pdf={}".format( svg_file ,pdf_file ) )
    # os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))



if __name__ == '__main__':
    main()