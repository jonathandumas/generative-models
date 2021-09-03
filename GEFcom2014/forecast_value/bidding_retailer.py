# -*- coding: UTF-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GEFcom2014 import pv_data, wind_data, load_data, read_file
from GEFcom2014.forecast_value import Planner
from GEFcom2014.utils import dump_file


def res_model(s_model: dict, n_s: int, i: int, prices: dict, s_obs: dict, curtail:bool=True,  soc_max:float=0):
    """
    Compute the planning and dispatching for a given set of senarios.
    :param s_model: PV, wind, and load scenarios from a model, s_model['pv'] is of shape (#TEST, 24, 100)
    :param n_s: number of scenarios to select for optimization <= 100.
    :param i: day index.
    :param prices:
    :param s_obs: PV, wind, and load observations for the corresponding day.
    :return: dispatching solution into a list in keuros
    """
    n_s = min(100, n_s)
    scenarios = dict()
    scenarios['PV'] = s_model['pv'][i,:,:n_s].transpose() # shape = (n_s, 24) for the day i
    scenarios['W'] = s_model['wind'][i,:,:n_s].transpose() # shape = (n_s, 24) for the day i
    scenarios['L'] = s_model['load'][i,:,:n_s].transpose() # shape = (n_s, 24) for the day i

    # planner = Planner_dad(scenarios=scenarios, prices=prices, curtail=curtail)
    planner = Planner(scenarios=scenarios, prices=prices, curtail=curtail, soc_max=soc_max)

    planner.solve()
    sol = planner.store_solution()

    dis = Planner(scenarios=s_obs, prices=prices, x=sol['x'], curtail=curtail, soc_max=soc_max)
    dis.solve()
    sol_dis = dis.store_solution()

    return [sol_dis['obj'] / 1000, sol_dis['dad_profit'] / 1000, sol_dis['short_penalty'] / 1000, sol_dis['long_penalty'] / 1000, sol_dis['x']]


def read_data(dir: str, name: str, TEST_size:int, pv_zone: int, wind_zone: int, id:dict=None):
    """
    Read scenarios for a given model.
    Scenarios shape = (24*TEST_size*n_zone, n_s)
    """

    if name == 'UMNN_M_' or name == 'AN_M_':
        # print(name, id)
        s_pv = read_file(dir=dir, name='scenarios_pv_' + name+str(id['pv'])+'_0_100_TEST')[24 * TEST_size * (pv_zone - 1):24 * TEST_size * pv_zone, :] # scenarios shape = (24*TEST_size, n_s)
        n_s = s_pv.shape[1]
        s_w = read_file(dir=dir, name='scenarios_wind_' + name+str(id['wind'])+'_0_100_TEST')[24 * TEST_size * (wind_zone - 1):24 * TEST_size * wind_zone, :] # scenarios shape = (24*TEST_size, n_s)
        s_l = read_file(dir=dir, name='scenarios_load_' + name+str(id['load'])+'_0_100_TEST') # scenarios shape = (24*TEST_size*n_zone, n_s)
    else:
        s_pv = read_file(dir=dir, name='scenarios_pv_' + name)[24 * TEST_size * (pv_zone - 1):24 * TEST_size * pv_zone, :]  # scenarios shape = (24*TEST_size, n_s)
        n_s = s_pv.shape[1]
        s_w = read_file(dir=dir, name='scenarios_wind_' + name)[24 * TEST_size * (wind_zone - 1):24 * TEST_size * wind_zone, :]  # scenarios shape = (24*TEST_size, n_s)
        s_l = read_file(dir=dir, name='scenarios_load_' + name)  # scenarios shape = (24*TEST_size*n_zone, n_s)

    s_model = dict()
    s_model['pv'] = s_pv.reshape(TEST_size, 24, n_s) # shape = (TEST_size, 24, n_s)
    s_model['wind'] = s_w.reshape(TEST_size, 24, n_s) # shape = (TEST_size, 24, n_s)
    s_model['load'] =  s_l.reshape(TEST_size, 24, n_s) # shape = (TEST_size, 24, n_s)
    return s_model

def bar_plot(dir: str, pdf_name: str, profit: np.array, penalty: np.array, net: np.array, labels: list, ylim: list, ylabel:str):
    """
    Bar plot of the results.
    """
    FONTSIZE = 12
    x = np.arange(len(labels))  # the label locations
    width = 0.33  # the width of the bars

    plt.figure()
    plt.bar(x - width, profit, width, label='profit')
    plt.bar(x, penalty, width, label='penalty')
    plt.bar(x + width, net, width, label='net')
    plt.ylabel(ylabel, fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(x, labels=labels, fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.xlim(-1, len(labels))
    plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()
    plt.savefig(dir + pdf_name + '.pdf')
    # plt.show()
    plt.close('all')


if __name__ == "__main__":

    print(os.getcwd())

    nb_s = 50
    soc_max = 1

    dir_path = 'export/bidding_retailer_'+str(nb_s)+'_'+str(int(100*soc_max))+'/'

    if not os.path.isdir(dir_path):  # test if directory exist
        os.makedirs(dir_path)

    # --------------------------------------------------------------------------------------------------------------
    # Day ahead spot and imbalance prices selection
    # --------------------------------------------------------------------------------------------------------

    # Load dad prices
    df_dad = pd.read_csv('../data/dad.csv', parse_dates=True, index_col=0)
    # select a day
    day_p = 6
    day_price = '2020-2-'+str(day_p)
    df_price =df_dad[day_price].values.reshape(-1)

    curtail = True
    # q_pos = q_neg = 2
    q_pos = 2
    q_neg = 2

    dad_price = 100  # euros /MWh
    # pos_imb = neg_imb = q * dad_price # euros /MWh
    pos_imb = q_pos * dad_price  # euros /MWh
    neg_imb = q_neg * dad_price  # euros /MWh
    gamma = (dad_price + pos_imb) / (pos_imb + neg_imb)
    print('dad_price %s pos_imb %s neg_imb %s GAMMA %s' % (dad_price, pos_imb, neg_imb, gamma))

    # prices = dict()
    # prices['dad'] = np.asarray([dad_price] * 24)
    # prices['imb +'] = np.asarray([pos_imb] * 24)
    # prices['imb -'] = np.asarray([neg_imb] * 24)

    prices = dict()
    # prices['dad'] = np.asarray([dad_price] * 16 + [3 * dad_price] * 4 + [dad_price] * 4) # peak during 4 hours
    prices['dad'] = df_price
    prices['imb +'] = q_pos * prices['dad']
    prices['imb -'] = q_neg * prices['dad']

    FONTSIZE = 10
    plt.figure()
    plt.plot(prices['dad'], label='dad price')
    # plt.plot(prices['imb -'] , label='imb neg')
    plt.ylabel('€/MWh', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dir_path+day_price+'_price.pdf')
    plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # GEFcom IJF_paper case study
    # Solar track: 3 zones
    # Wind track: 10 zones
    # Load track: 1 zones
    # 50 days picked randomly per zone for the VS and TEST sets
    # ------------------------------------------------------------------------------------------------------------------

    # WARNING: the time periods where PV is always 0 (night hours) are removed -> there are 8 periods removed
    # The index of the time periods removed are provided into indices
    data_pv, indices_pv = pv_data(path_name='../data/solar_new.csv', test_size=50, random_state=0)
    df_y_TEST_pv = data_pv[5].copy()
    non_null_indexes = list(np.delete(np.asarray([i for i in range(24)]), indices_pv))

    # Rebuilt the PV observations with the removed time periods
    df_y_TEST_pv.columns = non_null_indexes
    for k in indices_pv:
        df_y_TEST_pv[k] = 0
    df_y_TEST_pv = df_y_TEST_pv.sort_index(axis=1)

    data_wind = wind_data(path_name='../data/wind_data_all_zone.csv', test_size=50, random_state=0)
    df_y_TEST_wind = data_wind[5].copy()
    data_load = load_data(path_name='../data/load_data_track1.csv', test_size=50, random_state=0)
    df_y_TEST_load = data_load[5].copy()

    n_days_TEST = len(df_y_TEST_load)
    print('# TEST %s days' % (n_days_TEST))

    # random selection of the PV and wind zone
    zones_combinations = np.concatenate([np.asarray([[1, i] for i in range(1, 11)]), np.asarray([[2, i] for i in range(1, 11)]), np.asarray([[3, i] for i in range(1, 11)])], axis=0)
    # zones_combinations = np.asarray([[1, 1]])
    # pv_zone = np.random.randint(low=1, high=3, size=1)[0] # <= 3
    # wind_zone = np.random.randint(low=1, high=10, size=1)[0] # <= 10
    res_tot = []
    res_tot_all_comb = []
    res_profit_all_comb = []
    res_pen_all_comb = []
    labels = ['NF-UMNN', 'NF-A', 'VAE', 'GAN', 'GC', 'RAND', 'O']

    for comb, count in zip(range(zones_combinations.shape[0]), range(zones_combinations.shape[0])):

        pv_zone = zones_combinations[comb,0]
        wind_zone = zones_combinations[comb,1]

        print('%s pv_zone %s wind_zone %s' % (zones_combinations.shape[0] - count -1, pv_zone, wind_zone))

        df_y_TEST_pv_selected = df_y_TEST_pv.values[n_days_TEST * (pv_zone - 1):n_days_TEST * pv_zone, :].copy() # (50, 24)
        df_y_TEST_wind_selected = df_y_TEST_wind.values[n_days_TEST * (wind_zone - 1):n_days_TEST * wind_zone, :].copy() # (50, 24)

        # --------------------------------------------------------------------------------------------------------------
        # Load scenarios
        # --------------------------------------------------------------------------------------------------------------
        nf_a_id = {'pv': 10,
                   'wind': 8,
                   'load': 1}
        nf_umnn_id = {'pv': 3,
                      'wind': 1,
                      'load': 1}
        # scenarios are loaded into dict with each key having the shape (#TEST, 24, n_s)
        s_GC = read_data(dir='../forecast_quality/scenarios/gc/', name='gc_100_TEST', TEST_size=n_days_TEST, pv_zone=pv_zone, wind_zone=wind_zone)
        s_RAND = read_data(dir='../forecast_quality/scenarios/random/', name='random_100_TEST', TEST_size=n_days_TEST, pv_zone=pv_zone, wind_zone=wind_zone)
        s_UMNN = read_data(dir='../forecast_quality/scenarios/nfs/', name='UMNN_M_', TEST_size=n_days_TEST, pv_zone=pv_zone, wind_zone=wind_zone, id=nf_umnn_id)
        s_AN = read_data(dir='../forecast_quality/scenarios/nfs/', name='AN_M_', TEST_size=n_days_TEST, pv_zone=pv_zone, wind_zone=wind_zone, id=nf_a_id)
        s_VAE = read_data(dir='../forecast_quality/scenarios/vae/', name='VAElinear_1_0_100_TEST', TEST_size=n_days_TEST, pv_zone=pv_zone, wind_zone=wind_zone)
        s_GAN = read_data(dir='../forecast_quality/scenarios/gan/', name='GAN_wasserstein_1_0_100_TEST', TEST_size=n_days_TEST, pv_zone=pv_zone, wind_zone=wind_zone)

        # --------------------------------------------------------------------------------------------------------------
        # PLOT n days of the TEST set
        # --------------------------------------------------------------------------------------------------------------
        FONTSIZE = 10
        n_days = 1

        # plt.figure()
        # net = df_y_TEST_pv_selected[:n_days,:].reshape(n_days*24) + df_y_TEST_wind_selected[:n_days,:].reshape(n_days*24) - df_y_TEST_load.values[:n_days,:].reshape(n_days*24)
        # plt.plot(df_y_TEST_pv_selected[:n_days,:].reshape(n_days*24), label='PV')
        # plt.plot(df_y_TEST_wind_selected[:n_days,:].reshape(n_days*24), label='Wind')
        # plt.plot(df_y_TEST_load.values[:n_days,:].reshape(n_days*24), label='Load')
        # plt.plot(net.reshape(-1), '-D', label='net')
        # plt.ylim(-0.8, 1.2)
        # plt.xticks(fontsize=FONTSIZE)
        # plt.yticks(fontsize=FONTSIZE)
        # plt.legend(fontsize=1.5*FONTSIZE)
        # plt.tight_layout()
        # plt.savefig(dir_path +'test_set' + '_' + str(pv_zone) + '_' + str(wind_zone) +  '.pdf')
        # # plt.show()
        # plt.close('all')

        # Plots with different scales
        # Demonstrate how to do two plots on the same axes with different left and right scales.
        fig, ax1 = plt.subplots()
        # ax1.set_ylabel('MW')
        net = df_y_TEST_pv_selected[:n_days,:].reshape(n_days*24) + df_y_TEST_wind_selected[:n_days,:].reshape(n_days*24) - df_y_TEST_load.values[:n_days,:].reshape(n_days*24)
        ax1.plot(df_y_TEST_pv_selected[:n_days,:].reshape(n_days*24), label='PV')
        ax1.plot(df_y_TEST_wind_selected[:n_days,:].reshape(n_days*24), label='Wind')
        ax1.plot(df_y_TEST_load.values[:n_days,:].reshape(n_days*24), label='Load')
        ax1.plot(net.reshape(-1), '-D', label='net')
        ax1.tick_params(axis='y', labelsize=FONTSIZE)
        ax1.tick_params(axis='x', labelsize=FONTSIZE)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'black'
        ax2.set_ylabel('€/MWh', color=color)  # we already handled the x-label with ax1
        ax2.plot(prices['dad'], color=color, marker='P', label=r'$\pi_t$')
        ax2.tick_params(axis='y', labelcolor=color, labelsize=FONTSIZE)
        fig.legend(fontsize=1.5*FONTSIZE, ncol=5, bbox_to_anchor=(0., 0.95), loc='upper left') # The argument bbox_inches = "tight" to plt.savefig can be used to save the figure such that all artist on the canvas (including the legend) are fit into the saved area. If needed, the figure size is automatically adjusted.
        # fig.tight_layout()  #
        plt.savefig(dir_path +'test_set' + '_' + str(pv_zone) + '_' + str(wind_zone) +  '.pdf', bbox_inches="tight")
        # plt.show()
        plt.close('all')

        # ---------------------------------------------------------------------------------------------------
        # 4. Compute the bids using stochastic optimization

        res_UMNN = []
        res_AN = []
        res_VAE = []
        res_GAN = []
        res_GC = []
        res_RAND = []
        res_O = []
        tag_pdf = day_price + '_'  + str(pv_zone) + '_' + str(wind_zone)

        for day_i in range(n_days_TEST):
            print('%s' % (n_days_TEST - day_i - 1), end="\r", flush=True)

            # ----------------------------------------------------------------------------------------------------
            # 4. Compute the profit using the realization and the dad bids

            # Dispatch
            s_obs = dict()
            s_obs['PV'] = df_y_TEST_pv_selected[day_i].reshape(1, 24)
            s_obs['W'] = df_y_TEST_wind_selected[day_i].reshape(1, 24)
            s_obs['L'] = df_y_TEST_load.values[day_i].reshape(1, 24)

            sol_dis_UMNN = res_model(s_model=s_UMNN, n_s=nb_s, i=day_i, prices=prices, s_obs=s_obs, curtail=curtail, soc_max=soc_max)
            res_UMNN.append(sol_dis_UMNN)

            sol_dis_AN = res_model(s_model=s_AN, n_s=nb_s, i=day_i, prices=prices, s_obs=s_obs, curtail=curtail, soc_max=soc_max)
            res_AN.append(sol_dis_AN)

            sol_dis_VAE = res_model(s_model=s_VAE, n_s=nb_s, i=day_i, prices=prices, s_obs=s_obs, curtail=curtail, soc_max=soc_max)
            res_VAE.append(sol_dis_VAE)

            sol_dis_GAN = res_model(s_model=s_GAN, n_s=nb_s, i=day_i, prices=prices, s_obs=s_obs, curtail=curtail, soc_max=soc_max)
            res_GAN.append(sol_dis_GAN)

            sol_dis_GC = res_model(s_model=s_GC, n_s=nb_s, i=day_i, prices=prices, s_obs=s_obs, curtail=curtail, soc_max=soc_max)
            res_GC.append(sol_dis_GC)

            sol_dis_rand = res_model(s_model=s_RAND, n_s=nb_s, i=day_i, prices=prices, s_obs=s_obs, curtail=curtail, soc_max=soc_max)
            res_RAND.append(sol_dis_rand)

            # Oracle
            oracle = Planner(scenarios=s_obs, prices=prices, curtail=curtail, soc_max=soc_max)
            oracle.solve()
            sol_O = oracle.store_solution()
            res_O.append([sol_O['obj'] / 1000, sol_O['dad_profit'] / 1000, sol_O['short_penalty'] / 1000, sol_O['long_penalty'] / 1000])

            res_tot.append([sol_dis_UMNN[0], sol_dis_AN[0], sol_dis_VAE[0], sol_dis_GAN[0], sol_dis_GC[0], sol_dis_rand[0], sol_O['obj'] / 1000])

        # shape (n_days_TEST, )
        res_UMNN = np.asarray(res_UMNN)
        res_AN = np.asarray(res_AN)
        res_VAE = np.asarray(res_VAE)
        res_GAN = np.asarray(res_GAN)
        res_GC = np.asarray(res_GC)
        res_RAND = np.asarray(res_RAND)
        res_O = np.asarray(res_O)

        tot_pen_UMNN = res_UMNN[:, 2].sum() + res_UMNN[:, 3].sum()
        tot_pen_AN = res_AN[:, 2].sum() + res_AN[:, 3].sum()
        tot_pen_VAE =res_VAE[:, 2].sum() + res_VAE[:, 3].sum()
        tot_pen_GAN =res_GAN[:, 2].sum() + res_GAN[:, 3].sum()
        tot_pen_GC = res_GC[:, 2].sum() + res_GC[:, 3].sum()
        tot_pen_RAND = res_RAND[:, 2].sum() + res_RAND[:, 3].sum()
        tot_pen_O = res_O[:, 2].sum() + res_O[:, 3].sum()

        profit_UMNN = res_UMNN[:, 1].sum()
        profit_AN = res_AN[:, 1].sum()
        profit_VAE = res_VAE[:, 1].sum()
        profit_GAN = res_GAN[:, 1].sum()
        profit_GC = res_GC[:, 1].sum()
        profit_RAND = res_RAND[:, 1].sum()
        profit_O = res_O[:, 1].sum()

        tot_UMNN = res_UMNN[:, 0].sum()
        tot_AN = res_AN[:, 0].sum()
        tot_VAE = res_VAE[:, 0].sum()
        tot_GAN = res_GAN[:, 0].sum()
        tot_GC = res_GC[:, 0].sum()
        tot_RAND = res_RAND[:, 0].sum()
        tot_O = res_O[:, 0].sum()

        profit = np.asarray([profit_UMNN, profit_AN,  profit_VAE, profit_GAN, profit_GC,  profit_RAND,  profit_O])
        penalty = np.asarray([tot_pen_UMNN, tot_pen_AN, tot_pen_VAE, tot_pen_GAN, tot_pen_GC, tot_pen_RAND, tot_pen_O])
        net = np.asarray([tot_UMNN, tot_AN, tot_VAE, tot_GAN, tot_GC, tot_RAND, tot_O])

        pdf_name = 'results_' + tag_pdf + '_' + str(curtail)+ '_'+ str(int(100*q_neg))+ '_' +  str(int(100*q_pos))+ '_' + str(nb_s) + '_' + str(int(100*soc_max))
        bar_plot(dir=dir_path, pdf_name=pdf_name, profit=profit, penalty=penalty, net=net, labels=labels, ylim=[-15, 15], ylabel='k€')

        res_profit_all_comb.append([profit_UMNN, profit_AN,  profit_VAE, profit_GAN, profit_GC, profit_RAND,   profit_O])
        res_pen_all_comb.append([tot_pen_UMNN, tot_pen_AN,  tot_pen_VAE, tot_pen_GAN, tot_pen_GC,  tot_pen_RAND,  tot_pen_O])
        res_tot_all_comb.append([tot_UMNN, tot_AN ,  tot_VAE, tot_GAN, tot_GC , tot_RAND,   tot_O])

    # Dump results
    res_profit_all_comb = np.asarray(res_profit_all_comb)
    res_pen_all_comb = np.asarray(res_pen_all_comb)
    res_tot_all_comb = np.asarray(res_tot_all_comb)

    res_tot = np.asarray(res_tot)

    dump_file(dir=dir_path, name='res_tot_' + str(curtail)+ '_'+ str(int(100*q_neg))+ '_' +  str(int(100*q_pos)), file=res_tot)
    dump_file(dir=dir_path, name='res_profit_all_comb_' + str(curtail)+ '_'+ str(int(100*q_neg))+ '_' +  str(int(100*q_pos)), file=res_profit_all_comb)
    dump_file(dir=dir_path, name='res_pen_all_comb_' + str(curtail)+ '_'+ str(int(100*q_neg))+ '_' +  str(int(100*q_pos)), file=res_pen_all_comb)
    dump_file(dir=dir_path, name='res_tot_all_com_' + str(curtail)+ '_'+ str(int(100*q_neg))+ '_' +  str(int(100*q_pos)), file=res_tot_all_comb)


    res_tot = read_file(dir=dir_path, name='res_tot_' + str(curtail)+ '_'+ str(int(100*q_neg))+ '_' +  str(int(100*q_pos)))
    res_profit_all_comb = read_file(dir=dir_path, name='res_profit_all_comb_' + str(curtail)+ '_'+ str(int(100*q_neg))+ '_' +  str(int(100*q_pos)))
    res_pen_all_comb = read_file(dir=dir_path, name='res_pen_all_comb_' + str(curtail)+ '_'+ str(int(100*q_neg))+ '_' +  str(int(100*q_pos)))
    res_tot_all_comb = read_file(dir=dir_path, name='res_tot_all_com_' + str(curtail)+ '_'+ str(int(100*q_neg))+ '_' +  str(int(100*q_pos)))

    pdf_name = 'res_all_com_' + str(curtail) + '_' + str(int(100 * q_neg)) + '_' + str(int(100 * q_pos)) + '_' + str(nb_s) + '_' + str(int(100*soc_max))
    bar_plot(dir=dir_path, pdf_name=pdf_name, profit=res_profit_all_comb.sum(axis=0)/1000, penalty=res_pen_all_comb.sum(axis=0)/1000, net=res_tot_all_comb.sum(axis=0)/1000, labels=labels, ylim=[-0.3, 0.45], ylabel='M€')

    print('Net')
    print(labels)
    print(np.round(res_tot_all_comb.sum(axis=0),0))

    # Count the number of times which method is first
    tot_count = (-res_tot).argsort()
    # tot_count_reduced = (-res_tot[:,:-3]).argsort()
    tot_count_reduced = (-np.delete(res_tot[:,:-3], 1, axis=1)).argsort()

    ranking_final = []
    for j in range(len(['NF-UMNN', 'VAE', 'GAN'])):
        ranking = []
        for i in range(len(['NF-UMNN', 'VAE', 'GAN'])):
            ranking.append(sum(tot_count_reduced[:, j] == i))
        ranking_final.append(ranking)
    ranking_final = np.asarray(ranking_final)

    # Plot the cumulative rank for 'NF-UMNN', 'VAE', 'GAN'
    rank_percentage = 100 / tot_count_reduced.shape[0] * ranking_final

    for r, l in zip(range(rank_percentage.shape[0]), ['NF-UMNN', 'VAE', 'GAN']):
        print(l, np.round(np.cumsum(rank_percentage[:,r]),1))

    pdf_name = 'cum_rank_' + str(curtail) + '_' + str(int(100 * q_neg)) + '_' + str(int(100 * q_pos)) + '_' + str(nb_s) + '_' + str(int(100 * soc_max))
    FONTSIZE = 10
    x_index = [i for i in range(1, rank_percentage.shape[0]+1)]
    plt.figure()
    for r, l in zip(range(rank_percentage.shape[0]),['NF-UMNN', 'VAE', 'GAN']):
        plt.plot(x_index, np.cumsum(rank_percentage[:,r]), label=l, marker='D',linewidth=2)
    plt.hlines(y=100, xmin=1,xmax=3)
    plt.vlines(x=1, ymin=0,ymax=100)
    plt.ylim(25, 105)
    # plt.xlim(0, N_q + 1)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xticks(ticks=x_index)
    plt.xlabel('rank', fontsize=FONTSIZE)
    plt.ylabel('%', fontsize=FONTSIZE)
    plt.legend(fontsize=1.5 * FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_path + pdf_name+'.pdf')
    plt.show()

    # with all approaches

    ranking_final = []
    for j in range(len(labels)):
        ranking = []
        for i in range(len(labels)):
            ranking.append(sum(tot_count[:, j] == i))
        ranking_final.append(ranking)
    ranking_final = np.asarray(ranking_final)

    import seaborn as sns
    FONTSIZE = 12
    pdf_name = 'rank_' + str(curtail) + '_' + str(int(100 * q_neg)) + '_' + str(int(100 * q_pos)) + '_' + str(nb_s) + '_' + str(int(100 * soc_max))
    plt.figure(figsize=(6, 5))
    sns.set(font_scale=1.5)
    sns_plot = sns.heatmap(100 /tot_count.shape[0] * ranking_final[1:,:-1], cmap='RdYlGn_r', fmt=".1f", linewidths=0.5, xticklabels=True, yticklabels=True, annot=True, vmin=0, vmax=50, cbar_kws={'label': '%'}, annot_kws={"size": FONTSIZE})
    sns_plot.set_xticklabels(labels=labels[:-1], rotation='horizontal', fontsize=FONTSIZE)
    sns_plot.set_yticklabels(labels=[i for i in range(1,len(labels))], rotation='horizontal', fontsize=FONTSIZE)
    sns_plot.figure.axes[-1].yaxis.label.set_size(FONTSIZE)
    plt.tight_layout()
    plt.savefig(dir_path + pdf_name + '.pdf')
    plt.show()