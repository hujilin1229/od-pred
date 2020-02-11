import pandas as pd
import numpy as np
import datetime
import matplotlib.dates as md
from lib.plot import *
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from matplotlib import rc, font_manager
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams['xtick.major.pad'] = '8'
rcParams['xtick.minor.pad'] = '20'
rcParams['ytick.major.pad'] = '8'
rcParams['lines.linewidth'] = 2
rcParams['font.serif'] = 'Times New Roman'
rcParams['lines.markersize'] = 15
rcParams['legend.numpoints'] = 1
rcParams['lines.color'] = 'black'
rcParams["patch.force_edgecolor"] = True

sizeOfFont =22
fontProperties = {'family' : 'serif', 'serif':['Times New Roman'],
    'weight' : 'normal', 'size' : sizeOfFont}

rc('font',**fontProperties)

legend_font = {'family': 'serif',
        'weight': 'normal',
        'size': 20,
        }
bar_legend_font = {'family': 'serif',
        'weight': 'normal',
        'size': sizeOfFont,
        }
label_font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': sizeOfFont,
        }
sec_label_font = {'family': 'serif',
        'color':  'green',
        'weight': 'normal',
        'size': sizeOfFont,
        }
color_list = ['slategrey', 'darkviolet', 'red', 'k', 'b', 'r', 'c',
              'm', 'y', '#6ACC65',
              '#988ED5',  '#fdb462',
              '#feffb3', '#8b8b8b']
marker_list = ['.', 'v', 's', 'x', 'd']
line_style = ['ko-', 'r^-', 'bd-', 'rh-', 'm*-']
patterns = [ "ko", "\\" , " ", "\/", "-" , "*" , "x", "o", ".", "*" ]
line_width = 2.5
mark_size = 2
fig_dpi = 100
label_space = 0.1

def hist_figure():
    # zone = 'polygon'
    zone = 'taxi_zone'
    # df_file = './result/GODS_{}_S15_s6h3.csv'.format(zone)
    # df_fcd_file = './result/od_{}_S15_s6h3.csv'.format(zone)
    # df_fods_file = './result/direct_{}_S15_s6h3.csv'.format(zone)

    df_file = './result/GODS_{}_S15_s6h3_add_sigma_S3H3.csv'.format(zone)
    df_fcd_file = './result/od_{}_S15_s6h3_S3H3.csv'.format(zone)
    df_fods_file = './result/direct_{}_S15_s6h3_S3H3.csv'.format(zone)

    #
    df_fods = pd.read_csv(df_fcd_file, sep=',')
    df_fcd = pd.read_csv(df_fods_file, sep=',')
    df_data = pd.read_csv(df_file, sep=',')

    df_data['Interval'] = df_data['TI'] % 96
    df_fcd['Interval'] = df_fcd['TI'] % 96
    df_fods['Interval'] = df_fods['TI'] % 96

    df_mura_file = './result/MURA_OD_nyc_15.csv'
    df_mura = pd.read_csv(df_mura_file, sep=',')
    df_mura['Interval'] = df_fods['TI'] % 96

    print(df_data.head())
    df_dict = {
        'MR': df_mura,
        'RNN': df_fcd,
        'BF': df_fods,
        'AF': df_data[df_data.Hopk == 2]
    }

    for i, keyf in enumerate(['FC', 'BF', 'AF']):
        print(keyf)
        for ki, key in enumerate(['KL', 'JS', 'EMD']):
            x = []
            y = []
            for label, data in df_dict[keyf].groupby('Horizon'):
                x.append(label)
                y.append(data[key].mean())
            print(x)
            print(y)
            print("_________________________")

    # fig, ax = plt.subplots(1)
    # for label, data in df_data.groupby('Horizon'):
    #     for ki, key in enumerate(['KL', 'JS', 'EMD']):
    #         x = []
    #         y = []
    #         std = []
    #         for interval, data_int in data.groupby('Hopk'):
    #             x.append(interval)
    #             y.append(data_int[key].mean())
    #             std.append(data_int[key].std())
    #         plt.plot(x, y, label=key, color=color_list[ki],
    #                     marker=marker_list[ki], markersize=10)
    #     plt.legend(labelspacing=label_space, frameon=False)
    #     plt.ylim([0.0, 0.5])
    #     plt.xticks(range(2, 11, 2))
    #     plt.xlabel(r'$\alpha$', fontdict=label_font)
    #     plt.savefig('D:\\Dropbox\\OD_Prediction\\WWW19\\figure'
    #                     '\\Diff_metrics_hopk_{}_{}.pdf'.format(zone, label), dpi=1000)
    #     plt.show()
    #
    # # # # Plot the Performance over time
    # ylims = [0.4, 0.6, 0.4]
    # leg_loc = [(0.4, 0.6), (0.0, 0.15), (0., 0.25)]
    # for mi, method in enumerate(['EMD', 'KL', 'JS']):
    #     for label, data in df_data.groupby('Horizon'):
    #         fig, ax = plt.subplots(1)
    #         ax2 = ax.twinx()
    #         for i, key in enumerate(['FC', 'BF', 'AF']):
    #             df_i = df_dict[key]
    #             df_i = df_i[df_i.Horizon == label]
    #             total_amount = df_i.shape[0]
    #             x = []
    #             y = []
    #             time_x = []
    #             pers = []
    #             for interval in range(8):
    #                 df_i_j = df_i[(df_i.Interval >= interval* 12) & (df_i.Interval < (interval + 1)*12)]
    #                 if interval < 7:
    #                     time_x.append(datetime.datetime(
    #                         year=2013, month=8, day=3,
    #                         hour=int((interval+1)*3), minute=0))
    #                 else:
    #                     time_x.append(datetime.datetime(
    #                         year=2013, month=8, day=4,
    #                         hour=0, minute=0))
    #                 x.append(interval)
    #                 y.append(df_i_j[method].mean())
    #                 pers.append(df_i_j.shape[0] / total_amount * 100)
    #
    #             line_x = np.array(x)*3 + 1.5
    #             bar_x  = np.array(x)*3
    #             ax.plot(line_x, y, label=key, color=color_list[i],
    #                     marker=marker_list[i], markersize=10)
    #             ax2.bar(bar_x, pers, align='edge', width=3, hatch='//')
    #
    #         ax2.set_ylim([0.0, 50.0])
    #         ax2.set_ylabel('Percentage (%)', fontdict=sec_label_font)
    #         ax2.tick_params('y', colors='g')
    #
    #         ax.legend(loc=leg_loc[mi], labelspacing=label_space, frameon=False)
    #         ax.set_ylim([0.0, ylims[mi]])
    #         ax.set_ylabel(method, fontdict=label_font)
    #         ax.set_xlabel('Hour in a Day', fontdict=label_font)
    #
    #         xlocator = np.arange(0, 25, 3)
    #         ax.set_xticks(xlocator)
    #         ax.set_xlim([0.0, 24])
    #         # xformatter = md.DateFormatter('%H:%M')
    #         # xlocator = md.HourLocator(byhour=range(0, 24, 1), interval=6)
    #         # ax.xaxis.set_major_locator(xlocator)
    #         # ax.xaxis.set_major_formatter(xformatter)
    #         # datemin = datetime.datetime(year=2013, month=8, day=3, hour=2, minute=30)
    #         # ax.set_xlim(datemin)
    #
    #         plt.savefig('D:\\Dropbox\\OD_Prediction\\WWW19\\figure'
    #                     '\\Diff_methods_time_{}_{}_{}.pdf'.format(zone, method, label), dpi=1000)
    #         plt.show()
    #
    # # Plot the Performance over Distance
    # ylims = [0.4, 0.6, 0.5]
    # leg_loc = [(0.4, 0.6), (0.5, 0.2), (0.5, 0.2)]
    # for mi, method in enumerate(['EMD', 'KL', 'JS']):
    #     for label, data in df_data.groupby('Horizon'):
    #         fig, ax = plt.subplots(1)
    #         ax2 = ax.twinx()
    #         for i, key in enumerate(['FC', 'BF', 'AF']):
    #             df_i = df_dict[key]
    #             df_i = df_i[df_i.Horizon == label]
    #             dist_range = np.arange(0., 3.01, 0.5)
    #             df_i = df_i[df_i.Dist <= 3.1]
    #             total_amount = df_i.shape[0]
    #             x = []
    #             y = []
    #             pers = []
    #             for j, dist_i in enumerate(dist_range[:-1]):
    #                 df_i_j = df_i[(df_i.Dist >= dist_i) & (df_i.Dist < dist_range[j + 1])]
    #                 x.append(dist_i)
    #                 y.append(df_i_j[method].mean())
    #                 pers.append(df_i_j.shape[0] / total_amount * 100)
    #
    #             line_x = np.array(x) + 0.25
    #             bar_x = np.array(x)
    #             ax.plot(line_x, y, label=key, color=color_list[i],
    #                     marker=marker_list[i], markersize=10)
    #             ax2.bar(bar_x, pers, align='edge', width=0.5, hatch='//')
    #
    #         ax2.set_ylim([0.0, 90.0])
    #         ax2.set_ylabel('Percentage (%)', fontdict=sec_label_font)
    #         ax2.tick_params('y', colors='g')
    #
    #         xlocator = np.arange(0., 3.01, 0.5)
    #         ax.set_xticks(xlocator)
    #         ax.set_xlim([0.0, 3.0])
    #
    #         ax.legend(loc=leg_loc[mi], labelspacing=label_space, frameon=False)
    #         ax.set_ylim([0.0, ylims[mi]])
    #         ax.set_ylabel(method, fontdict=label_font)
    #         ax.set_xlabel('Distance (km)', fontdict=label_font)
    #         plt.savefig('D:\\Dropbox\\OD_Prediction\\WWW19\\figure'
    #                     '\\Diff_methods_dist_{}_{}_{}.pdf'.format(zone, method, label), dpi=1000)
    #         plt.show()

    # # Plot the Data Distribution
    # label_font = {'family': 'serif',
    #               'color': 'black',
    #               'weight': 'normal',
    #               'size': 18,
    #               }
    # data_file = './data/chengdu/SecRing/2014-08-03_2014-08-31.csv'
    # df_data = pd.read_csv(data_file)
    # fig, ax = plt.subplots()
    # ax2 = ax.twinx()
    # n, bins, patches = ax.hist(df_data.speed, bins=10, density=True)
    # n, bins, patches = ax2.hist(
    #     df_data.speed, cumulative=1, histtype='step', bins=10,
    #     color='tab:orange', density=True)
    # plt.xlim([0.0, 18.0])
    # plt.xticks(range(0, 19, 2))
    # ax.set_ylabel('PDF', fontdict=label_font)
    # ax2.set_ylabel('CDF', fontdict=label_font)
    # plt.xlabel('Speed (m/s)', fontdict=label_font)
    # plt.savefig('CD_Speed_Hist.pdf', dpi=1000)
    #
    # data_file = './data/nyc/Manhattan/2013-11-01_2014-01-01.csv'
    # df_data = pd.read_csv(data_file)
    # fig, ax = plt.subplots()
    # ax2 = ax.twinx()
    # data_speed = df_data.speed * 1607
    # n, bins, patches = ax.hist(data_speed, bins=10, density=True)
    # n, bins, patches = ax2.hist(
    #     data_speed, cumulative=1, histtype='step', bins=10,
    #     color='tab:orange', density=True)
    # plt.xlim([0.0, 21.0])
    # plt.xticks(range(0, 22, 4))
    # ax.set_ylabel('PDF', fontdict=label_font)
    # ax2.set_ylabel('CDF', fontdict=label_font)
    # plt.xlabel('Speed (m/s)', fontdict=label_font)
    # plt.savefig('NYC_Speed_Hist.pdf', dpi=1000)






if __name__ == '__main__':
    hist_figure()