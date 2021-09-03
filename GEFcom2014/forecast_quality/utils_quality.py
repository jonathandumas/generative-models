# -*- coding: utf-8 -*-

import math
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from GEFcom2014 import read_file

def crps_nrg_marginal(y_true:float, y_sampled:np.array):
    """
    Compute the CRPS NRG for a given time period k -> the CRPS is a univariate metric.
    Therefore it has to be computed per marginal.
    :param y_true: true value for this time period.
    :param y_sampled: nb quantile/scenarios for time period k with shape(nb,)
    """
    nb = y_sampled.shape[0] # Nb of quantiles/scenarios sampled.
    simple_sum = np.sum(np.abs(y_sampled - y_true)) / nb
    double_somme = 0
    for i in range(nb):
        for j in range(nb):
            double_somme += np.abs(y_sampled[i] - y_sampled[j])
    double_sum = double_somme / (2 * nb * nb)

    crps = simple_sum  - double_sum

    return crps

def plf_per_quantile(quantiles:np.array, y_true:np.array):
    """
    Compute PLF per quantile.
    :param quantiles: (nb_periods, nb_quantiles)
    :param y_true:  (nb_periods,)
    :return: PLF per quantile into an array (nb_quantiles, )
    """
    # quantile q from 0 to N_q -> add 1 to be from 1 to N_q into the PLF score
    N_q = quantiles.shape[1]
    plf = []
    for q in range(0 ,N_q):
        # for a given quantile compute the PLF over the entire dataset
        diff = y_true - quantiles[:,q]
        plf_q = sum(diff[diff >= 0] * ((q+1) / (N_q+1))) / len(diff) + sum(-diff[diff < 0] * (1 - (q+1) / (N_q+1))) / len(diff) # q from 0 to N_q-1 -> add 1 to be from 1 to N_q
        plf.append(plf_q)
    return 100 * np.asarray(plf)

def plot_plf_per_quantile(plf_VS: np.array, plf_TEST: np.array, dir_path: str, name: str, ymax:float=None):
    """
    Plot the quantile score (PLF = Pinball Loss Function) per quantile on the VS & TEST sets.
    """
    FONTSIZE = 10
    plt.figure()
    plt.plot([q for q in range(1, len(plf_VS) + 1)], plf_TEST, 'b')
    plt.hlines(y=plf_TEST.mean(), colors='b', xmin=1, xmax=len(plf_VS),  label='TEST av ' + str(round(plf_TEST.mean(), 4)))
    plt.plot([q for q in range(1, len(plf_VS) + 1)], plf_VS, 'g')
    plt.hlines(y=plf_VS.mean(), colors='g', xmin=1, xmax=len(plf_VS), label='VS av ' + str(round(plf_VS.mean(), 4)))
    if ymax:
        plt.ylim(0, ymax)
        plt.vlines(x=(len(plf_VS) + 1) / 2, colors='k', ymin=0, ymax=ymax)
    else:
        plt.ylim(0, max(plf_TEST.max(), plf_VS.max()))
        plt.vlines(x=(len(plf_VS) + 1) / 2, colors='k', ymin=0, ymax=max(plf_TEST.max(), plf_VS.max()))
    plt.xlim(0, len(plf_VS) + 1)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlabel('q', fontsize=FONTSIZE)
    plt.ylabel('%', fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_path + name + '.pdf')
    plt.show()

def compare_plf(dir: str, plf: list, name: str, ylim:list, labels:str):
    """
    Plot the quantile score (PLF = Pinball Loss Function) per quantile on TEST sets of multiple generative models.
    :param plf: list of the plf_score of multiple generative models. Each element of the list is an array.
    """
    x_index = [q for q in range(1, len(plf[0]) + 1)]
    FONTSIZE = 10
    plt.figure()
    for l, label in zip(plf, labels):
        plt.plot(x_index, l, label=label)
    plt.ylim(ylim[0], ylim[1])
    plt.vlines(x=(len(plf[0])  + 1) / 2, colors='k', ymin=0, ymax=ylim[1])
    plt.xlim(0, len(plf[0])  + 1)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlabel('$q$', fontsize=FONTSIZE)
    plt.ylabel('%', fontsize=FONTSIZE)
    plt.legend(fontsize=1.5*FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir + name + '.pdf')
    plt.show()

def compute_reliability(y_true: np.array, y_quantile: np.array, tag: str = 'pv'):
    """
    Compute averaged reliability score per day over all quantiles.
    :param y_true: true values (n_periods, ).
    :param y_quantile: quantiles (n_periods, n_quantiles).
    :return: PLF array of shape (n_quantiles,)
    """
    nb_q = y_quantile[0].shape[0]

    aq = []
    if tag == 'pv':
        # WARNING REMOVE TIME PERIODS WHERE PV GENERATION IS 0 during night hours !!!!
        # indices where PV generation is 0 at day d
        indices = np.where(y_true == 0)[0]
        y_true = np.delete(y_true, indices).copy()
        y_quantile = np.delete(y_quantile, indices, axis=0).copy()

    nb_periods = len(y_true)
    for q in range(0, nb_q):
        aq.append(sum(y_true < y_quantile[:, q]) / nb_periods)

    return 100 * np.asarray(aq)


def reliability_diag(aq_VS: np.array, aq_TEST: np.array, dir_path: str, name: str):
    """
    Reliablity diagram per quantile on the VS & TEST sets.
    :param aq_VS:  (n_quantiles, ).
    :param aq_TEST:  (n_quantiles, ).
    """

    N_q = aq_VS.shape[0]
    q_set = [i / (N_q + 1) for i in range(1, N_q + 1)]

    FONTSIZE = 10
    x_index = np.array(q_set) * 100
    plt.figure()
    plt.plot(x_index, x_index, 'k')
    plt.plot(x_index, aq_TEST, 'b', label='TEST')
    plt.plot(x_index, aq_VS, 'g', label='VS')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('$q$', fontsize=FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xticks(ticks=[i for i in range(0, 100 + 10, 10)])
    plt.yticks(ticks=[i for i in range(0, 100 + 10, 10)])
    plt.ylabel('%', fontsize=FONTSIZE)
    plt.title(name)
    plt.legend(fontsize=FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_path + name + '.pdf')
    plt.show()


def plot_multi_days_scenarios(quantiles: np.array, scenarios: np.array, y_true: np.array, dir_path: str, name: str, n_s: int,
                              ylim: float = 1):
    """
    Plot 4 days with subplots (2, 1).
    :param quantiles: (n_periods, n_s).
    :param scenarios: (n_periods, n_q).
    :param y_true: np.array with (n_periods, )
    """

    FONTSIZE = 10
    x_index = [i for i in range(1, 2 * 24 + 1)]
    fig, axs = plt.subplots(2, 1, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
    axs[0].plot(x_index, scenarios[:2 * 24, :n_s], color='gray', linewidth=1, alpha=0.5)
    axs[0].plot(x_index, quantiles[:2 * 24, 9], color='b', linewidth=2, label='10 %')
    axs[0].plot(x_index, quantiles[:2 * 24, 49], color='k', linewidth=2, label='50 %')
    axs[0].plot(x_index, quantiles[:2 * 24, 89], color='g', linewidth=2, label='90 %')
    axs[0].plot(x_index, y_true[:2 * 24], color='r', linewidth=2, label='obs')
    axs[0].set_ylim(0, ylim)
    axs[0].tick_params(axis='both', labelsize=FONTSIZE)

    axs[1].plot(x_index, scenarios[2 * 24:4 * 24, :n_s], color='gray', linewidth=1, alpha=0.5)
    axs[1].plot(x_index, quantiles[2 * 24:4 * 24, 9], color='b', linewidth=2, label='10 %')
    axs[1].plot(x_index, quantiles[2 * 24:4 * 24, 49], color='k', linewidth=2, label='50 %')
    axs[1].plot(x_index, quantiles[2 * 24:4 * 24, 89], color='g', linewidth=2, label='90 %')
    axs[1].plot(x_index, y_true[2 * 24:4 * 24], color='r', linewidth=2, label='obs')
    axs[1].set_ylim(0, ylim)
    axs[1].tick_params(axis='both', labelsize=FONTSIZE)

    for ax in axs.flat:
        ax.label_outer()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:4], labels[:4], fontsize=FONTSIZE, loc='center right', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(dir_path + name + '.pdf')
    plt.show()

def plot_scenarios(quantiles: np.array, scenarios: np.array, y_true: np.array, dir_path: str, name: str, n_s: int, n_days:int=1, ylim:list=[0, 1]):
    """
    Plot n_days days in one plot consecutively.
    :param quantiles: (n_periods, n_s).
    :param scenarios: (n_periods, n_q).
    :param y_true: np.array with (n_periods, )
    """
    FONTSIZE = 15
    x_index = [i for i in range(1, n_days * 24 + 1)]
    plt.figure()
    plt.plot(x_index, scenarios[:n_days * 24, :n_s], color='gray', linewidth=1, alpha=0.5)
    plt.plot(x_index, quantiles[:n_days * 24, 9], color='b', linewidth=2, label='10 %')
    plt.plot(x_index, quantiles[:n_days * 24, 49], color='k', linewidth=2, label='50 %')
    plt.plot(x_index, quantiles[:n_days * 24, 89], color='g', linewidth=2, label='90 %')
    plt.plot(x_index, y_true[:n_days * 24], color='r', linewidth=2, label='obs')
    plt.ylim(ylim[0], ylim[1])
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    legend = plt.legend(fontsize=1.5*FONTSIZE, loc='center right', borderaxespad=0.)
    legend.remove()
    plt.tight_layout()
    plt.savefig(dir_path + name + '.pdf')
    plt.show()


def crps_per_period(scenarios: np.array, y_true: np.array, max_s: int = 100):
    """
    Compute the CRPS per period or marginal. Indeed, the CRPS is a univariate score.
    :param scenarios: of shape (n_periods, n_s) with  n_periods = n_d * 24
    :param y_true: observations of shape (n_periods, ) with  n_periods = n_d * 24
    :return: averaged CRPS per time period CRPS into a np.array of shape (24,)
    """
    n_s = scenarios.shape[1]
    n_periods = scenarios.shape[0]
    n_d = int(len(y_true) / 24)
    # compute the CRPS with at most 100 scenarios otherwise take too much time and the difference over 100 is small
    n_s_max = min(n_s, max_s)

    # compute the CRPS over the entire TEST/VS set
    crps_t = np.asarray([crps_nrg_marginal(y_true=y_true[t], y_sampled=scenarios[t, :n_s_max]) for t in range(n_periods)])
    crps_averaged = crps_t.reshape(n_d, 24).mean(axis=0)  # averaged CRPS per time period (or by marginal)

    return crps_averaged, crps_t.reshape(n_d, 24)


def plot_crps_per_period(dir: str, name: str, crps_TEST: np.array, crps_VS: np.array, ylim: list, max_s: int):
    """
    :param crps_TEST: CRPS into a np.array of shape (24,)
    :param crps_VS: CRPS into a np.array of shape (24,)
    """
    TEST_av = 100 * crps_TEST.mean()
    VS_av = 100 * crps_VS.mean()

    FONTSIZE = 10
    plt.figure()
    plt.plot(100 * crps_TEST, 'b', label='TEST av = ' + str(round(TEST_av, 2)))
    plt.plot(100 * crps_VS, 'g', label='VS av = ' + str(round(VS_av, 2)))
    plt.hlines(y=TEST_av, xmin=0, xmax=23, colors='b')
    plt.hlines(y=VS_av, xmin=0, xmax=23, colors='g')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.ylabel('%', fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()
    plt.savefig(dir + name + '_crps_' + str(max_s) + '.pdf')
    plt.show()


def quantiles_and_evaluation(dir_path: str, s_VS: np.array, s_TEST: np.array, N_q: int, df_y_VS: pd.DataFrame,
                             df_y_TEST: pd.DataFrame, name: str, ymax_plf: float, ylim_crps: list, tag: str,
                             nb_zones: int):
    """
    Compute quantiles from scenarios.
    Compute metrics:
    - PLF
    - reliability diagrams
    - CRPS
    - plots of scenarios & quantiles

    #VS or #TEST = number of days of VS or TEST sets

    :param s_VS: array (#VS*24, n_s)
    :param s_TEST: array (#TEST*24, n_s)
    :param N_q: number of quantiles/percentiles.
    :param df_y_VS: VS true values into a pd.DataFrame with shape (#VS, 24)
    :param df_y_TEST: TEST true values into a pd.DataFrame with shape (#TEST, 24)
    :param name: name to export plots
    :param tag: pv, wind, load
    :param nb_zones: number of zones (PV 3, wind 10, load 1)
    """
    # --------------------------------------------------------------------------------------------------------------
    # 1. Generate quantiles from scenarios
    # --------------------------------------------------------------------------------------------------------------

    n_s = s_VS.shape[1]

    q_set = [i / (N_q + 1) for i in range(1, N_q + 1)]
    # Quantiles are generated into an array of shape (n_day*24, N_q)
    q_TEST = np.quantile(s_TEST, q=q_set, axis=1).transpose()
    q_VS = np.quantile(s_VS, q=q_set, axis=1).transpose()

    # --------------------------------------------------------------------------------------------------------------
    # 2. PLF TEST & VS
    # --------------------------------------------------------------------------------------------------------------

    plf_TEST = plf_per_quantile(quantiles=q_TEST, y_true=df_y_TEST.values.reshape(-1))
    plf_VS = plf_per_quantile(quantiles=q_VS, y_true=df_y_VS.values.reshape(-1))
    print('PLF TEST %.4f VS %.4f' % (plf_TEST.mean(), plf_VS.mean()))
    print('')

    plot_plf_per_quantile(plf_VS=plf_VS, plf_TEST=plf_TEST, dir_path=dir_path, name='plf_' + name + '_' + str(n_s),
                          ymax=ymax_plf)

    # --------------------------------------------------------------------------------------------------------------
    # 3. Reliability diagram
    # --------------------------------------------------------------------------------------------------------------
    nb_days_VS = len(df_y_VS)
    nb_days_TEST = len(df_y_TEST)

    aq_TEST = compute_reliability(y_true=df_y_TEST.values.reshape(-1), y_quantile=q_TEST, tag=tag)
    aq_VS = compute_reliability(y_true=df_y_VS.values.reshape(-1), y_quantile=q_VS, tag=tag)
    reliability_diag(aq_VS=aq_VS, aq_TEST=aq_TEST, dir_path=dir_path, name='reliability_' + name + '_' + str(n_s))

    mae_TEST = mean_absolute_error(y_true=np.array(q_set) * 100, y_pred=aq_TEST)
    mae_VS = mean_absolute_error(y_true=np.array(q_set) * 100, y_pred=aq_VS)
    rmse_TEST = math.sqrt(mean_squared_error(y_true=np.array(q_set) * 100, y_pred=aq_TEST))
    rmse_VS = math.sqrt(mean_squared_error(y_true=np.array(q_set) * 100, y_pred=aq_VS))
    print('MAE TEST %.2f VS %.2f' % (mae_TEST, mae_VS))
    print('RMSE TEST %.2f VS %.2f' % (rmse_TEST, rmse_VS))
    print('')

    # --------------------------------------------------------------------------------------------------------------
    # 4. PLOT 50 scenarios on the first 4 days of the TEST set over all the zones
    # --------------------------------------------------------------------------------------------------------------
    nb_days_per_zone = int(nb_days_TEST / nb_zones)
    # WARNING only for the first zone
    for z in range(nb_zones):
        plot_multi_days_scenarios(quantiles=q_TEST[24 * nb_days_per_zone * z:24 * nb_days_per_zone * (z + 1), :],
                                  scenarios=s_TEST[24 * nb_days_per_zone * z:24 * nb_days_per_zone * (z + 1), :],
                                  y_true=df_y_TEST.values.reshape(-1)[
                                  24 * nb_days_per_zone * z:24 * nb_days_per_zone * (z + 1)], dir_path=dir_path,
                                  name='TEST_zone_' + str(z) + '_' + name, n_s=50, ylim=1)
    # --------------------------------------------------------------------------------------------------------------
    # CRPS
    # --------------------------------------------------------------------------------------------------------------
    max_s = 100
    crps_TEST, crps_d_TEST = crps_per_period(scenarios=s_TEST, y_true=df_y_TEST.values.reshape(-1), max_s=max_s)
    crps_VS, crps_d_VS = crps_per_period(scenarios=s_VS, y_true=df_y_VS.values.reshape(-1), max_s=max_s)

    print('CRPS TEST %.2f VS %.2f' % (100 * crps_TEST.mean(), 100 * crps_VS.mean()))
    print('')

    plot_crps_per_period(dir=dir_path, name=name, crps_TEST=crps_TEST, crps_VS=crps_VS, ylim=ylim_crps, max_s=max_s)

def plot_corr_heat_map(dir: str, name: str, df_corr: pd.DataFrame, vlim: list):
    """
    Plot the correlation matrix with a heatmap
    cmap with _r = reverses the normal order of the color map 'RdYlGn'
    """
    FONTSIZE =15
    plt.figure()
    sns_plot = sns.heatmap(df_corr, cmap='RdYlGn_r', fmt=".1f", linewidths=0.5, xticklabels=False, yticklabels=False,
                           annot=False, vmin=vlim[0], vmax=vlim[1],cbar=False)
    sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation='horizontal')
    cb = sns_plot.figure.colorbar(sns_plot.collections[0])  # Display colorbar
    cb.ax.tick_params(labelsize=FONTSIZE)  # Set the colorbar scale font size.
    plt.tight_layout()
    plt.savefig(dir + name + '_corr.pdf')
    plt.show()

def quality_comparison_per_track(dir_path: str, N_q: int, df_y: pd.DataFrame, name: str, ylim_plf: list, ylim_crps: list, tag: str, crps:bool=True):
    """
    Compare the generative models with quality scores:
    - PLF
    - reliability diagrams
    - CRPS
    - plots of scenarios & quantiles

    :param N_q: number of quantiles/percentiles.
    :param df_y: true values into a pd.DataFrame with shape (#TEST, 24) where #TEST = number of days of the TEST set
    :param name: name to export plots
    :param tag: pv, wind, load
    """
    nf_a_id = {'pv': 10,
               'wind': 8,
               'load': 1}
    nf_umnn_id = {'pv': 3,
               'wind': 1,
               'load': 1}

    y_true = df_y.values.reshape(-1) # reshape from (#TEST, 24) to (24*,#TEST)
    # --------------------------------------------------------------------------------------------------------------
    # 0. Load scenarios on the TS for 'NF-UMNN', 'NF-A', 'VAE', 'GAN', 'GC', 'RAND'
    # --------------------------------------------------------------------------------------------------------------
    labels = ['NF-UMNN', 'NF-A', 'VAE', 'GAN', 'GC', 'RAND']
    # scenarios shape = (24*n_days, n_s)
    s_umnn = read_file(dir='scenarios/nfs/', name='scenarios_' + tag + '_UMNN_M_' + str(nf_umnn_id[tag]) + '_0_100_TEST')
    s_an = read_file(dir='scenarios/nfs/', name='scenarios_' + tag + '_AN_M_' + str(nf_a_id[tag]) + '_0_100_TEST')
    s_vae = read_file(dir='scenarios/vae/', name='scenarios_' + tag + '_VAElinear_1_0_100_TEST')
    s_gan = read_file(dir='scenarios/gan/', name='scenarios_' + tag + '_GAN_wasserstein_1_0_100_TEST')
    s_gc = read_file(dir='scenarios/gc/', name='scenarios_' + tag + '_gc_100_TEST')
    s_rand = read_file(dir='scenarios/random/', name='scenarios_' + tag + '_random_100_TEST')
    scenarios_list = [s_umnn, s_an, s_vae, s_gan, s_gc, s_rand]

    # --------------------------------------------------------------------------------------------------------------
    # 1. Generate quantiles from scenarios
    # --------------------------------------------------------------------------------------------------------------

    q_set = [i / (N_q + 1) for i in range(1, N_q + 1)]
    # Quantiles are generated into an array of shape (n_day*24, N_q), the same shape than scenarios
    quantiles_list = []
    for s in scenarios_list:
        quantiles_list.append(np.quantile(s, q=q_set, axis=1).transpose())

    # --------------------------------------------------------------------------------------------------------------
    # 2. PLF TEST & VS
    # --------------------------------------------------------------------------------------------------------------
    plf_list = []

    for q in quantiles_list:
        plf_list.append(plf_per_quantile(quantiles=q, y_true=y_true))

    print('PLF TS UMNN %.2f AN %.2f VAE %.2f GAN %.2f GC %.2f rand %.2f' % (plf_list[0].mean(), plf_list[1].mean(), plf_list[2].mean(), plf_list[3].mean(), plf_list[4].mean(), plf_list[5].mean()))
    print('')

    compare_plf(dir=dir_path, plf=plf_list, name='plf_' + name, ylim=ylim_plf, labels=labels)

    # --------------------------------------------------------------------------------------------------------------
    # 3. Reliability diagram
    # --------------------------------------------------------------------------------------------------------------
    aq_list = []
    for q in quantiles_list:
        aq_list.append(compute_reliability(y_true=y_true, y_quantile=q, tag=tag))

    compare_reliability_diag(dir=dir_path, aq=aq_list, name='reliability_' + name, labels=labels)
    mae_list = []
    for a in aq_list:
        mae_list.append(mean_absolute_error(y_true=np.array(q_set) * 100, y_pred=a))

    print('MAE TS UMNN %.2f AN %.2f VAE %.2f GAN %.2f GC %.2f rand %.2f' % (mae_list[0], mae_list[1], mae_list[2], mae_list[3], mae_list[4], mae_list[5]))
    print('')

    # --------------------------------------------------------------------------------------------------------------
    # CRPS
    # --------------------------------------------------------------------------------------------------------------
    if crps:
        max_s = 100
        crps_list = []
        for s in scenarios_list:
            crps, crps_d = crps_per_period(scenarios=s, y_true=y_true, max_s=max_s)
            crps_list.append(crps)

        print('CRPS TEST UMNN %.2f AN %.2f VAE %.2f GAN %.2f GC %.2f rand %.2f' % (100*crps_list[0].mean(), 100*crps_list[1].mean(), 100*crps_list[2].mean(), 100*crps_list[3].mean(), 100*crps_list[4].mean(), 100*crps_list[5].mean()))
        print('')

        compare_crps(dir=dir_path, crps=crps_list, name='crps_' + name, ylim=ylim_crps, labels=labels)

    # --------------------------------------------------------------------------------------------------------------
    # 4. PLOT 50 scenarios only over the first zone for PV and wind
    # --------------------------------------------------------------------------------------------------------------
    for q, s, model in zip(quantiles_list, scenarios_list, labels):
        plot_multi_days_scenarios(quantiles=q, scenarios=s, y_true=y_true, dir_path=dir_path, name='TEST_zone_1_multi_' + name + '_' + model, n_s=50, ylim=1)
        if tag == 'load':
            ylim = [0.2, 0.9]
        else:
            ylim = [0, 1]
        plot_scenarios(quantiles=q, scenarios=s, y_true=y_true, dir_path=dir_path, name='TEST_zone_1_' + name + '_' + model, n_s=50, n_days=1, ylim=ylim)


def compare_reliability_diag(dir: str, aq: list, name: str, labels:str):
    """
    Reliablity diagram per quantile.
    :param aq: list of the aq scores of multiple generative models. Each element of the list is an array of shape (n_q,).
    """

    N_q = aq[0].shape[0]
    q_set = [i / (N_q + 1) for i in range(1, N_q + 1)]
    x_index = np.array(q_set) * 100

    FONTSIZE = 10
    plt.figure()
    plt.plot(x_index, x_index, 'k', linewidth=2)
    for a, label in zip(aq, labels):
        plt.plot(x_index, a, label=label)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('$q$', fontsize=FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xticks(ticks=[i for i in range(0, 100 + 10, 10)])
    plt.yticks(ticks=[i for i in range(0, 100 + 10, 10)])
    plt.ylabel('%', fontsize=FONTSIZE)
    # plt.title(name)
    plt.legend(fontsize=1.5*FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir + name + '.pdf')
    plt.show()

def compare_crps(dir: str, crps: list, name: str, ylim:list, labels:str):
    """
    :param crps: list of the crps scores of multiple generative models. Each element of the list is an array.
    """

    FONTSIZE = 10
    plt.figure()
    for c, label in zip(crps, labels):
        plt.plot(100 * c, label=label)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.ylabel('%', fontsize=FONTSIZE)
    plt.legend(fontsize=1.5*FONTSIZE)
    plt.grid(True)
    plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()
    plt.savefig(dir + name + '.pdf')
    plt.show()

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
