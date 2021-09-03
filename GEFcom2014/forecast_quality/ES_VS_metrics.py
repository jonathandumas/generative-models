# -*- coding: utf-8 -*-

"""
Multivariate metric: ES and VS.
Then, the DM test is performed.
"""

import math
import os
import pickle
import random

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from GEFcom2014 import read_file
from GEFcom2014 import pv_data, wind_data, load_data, read_file
from GEFcom2014.forecast_quality import compute_DM


def energy_score(s: np.array, y_true: np.array):
    """
    Compute the Energy score (ES).
    :param s: scenarios of shape (24*n_days, n_s)
    :param y_true: observations of shape = (n_days, 24)
    :return: the ES per day of the testing set.
    """
    n_periods = y_true.shape[1]
    n_d = len(y_true)  # number of days
    n_s = s.shape[1]  # number of scenarios per day
    es = []
    # loop on all days
    for d in range(n_d):
        # select a day for both the scenarios and observations
        s_d = s[n_periods * d:n_periods * (d + 1), :]
        y_d = y_true[d, :]

        # compute the part of the ES
        simple_sum = np.mean([np.linalg.norm(s_d[:, s] - y_d) for s in range(n_s)])

        # compute the second part of the ES
        double_somme = 0
        for i in range(n_s):
            for j in range(n_s):
                double_somme += np.linalg.norm(s_d[:, i] - s_d[:, j])
        double_sum = double_somme / (2 * n_s * n_s)

        # ES per day
        es_d = simple_sum - double_sum
        es.append(es_d)
    return es

def variogram_score(s: np.array, y_true: np.array, beta: float):
    """
    Compute the Variogram score (VS).
    :param s: scenarios of shape (24*n_days, n_s)
    :param y_true: observations of shape = (n_days, 24)
    :param beta: order of the VS
    :return: the VS per day of the testing set.
    """
    n_periods = y_true.shape[1]
    n_d = len(y_true)  # number of days
    n_s = s.shape[1]  # number of scenarios per day
    weights = 1  # equal weights across all hours of the day
    vs = []
    # loop on all days
    for d in range(n_d):
        # select a day for both the scenarios and observations
        s_d = s[n_periods * d:n_periods * (d + 1), :]
        y_d = y_true[d, :]

        # Double loop on time periods of the day
        vs_d = 0
        for k1 in range(n_periods):
            for k2 in range(n_periods):
                # VS first part
                first_part = np.abs(y_d[k1] - y_d[k2]) ** beta
                second_part = 0
                # Loop on all scenarios to compute VS second part
                for i in range(n_s):
                    second_part += np.abs(s_d[k1, i] - s_d[k2, i]) ** beta
                second_part = second_part / n_s
                vs_d += weights * (first_part - second_part) ** 2
        # VS per day
        vs.append(vs_d)
    return vs


def plot_DM(p_value: np.array, dir: str, pdf_name: str):
    """
    Plot the DM test.
    """
    FONTSIZE = 20
    plt.figure()
    sns.set(font_scale=1.5)
    sns_plot = sns.heatmap(100 * p_value, cmap='RdYlGn_r', fmt=".1f", linewidths=0.5, xticklabels=True,
                           yticklabels=True, annot=False, vmin=0, vmax=10, annot_kws={"size": FONTSIZE})
    sns_plot.set_xticklabels(labels=models, rotation='horizontal', fontsize=FONTSIZE)
    sns_plot.set_yticklabels(labels=models, rotation=90, fontsize=FONTSIZE)
    sns_plot.figure.axes[-1].yaxis.label.set_size(FONTSIZE)
    plt.tight_layout()
    plt.savefig(dir + pdf_name + '.pdf')
    plt.show()

if __name__ == '__main__':

    """
    Energy and Variogram scores.
    """
    beta = 0.5 # VS order

    dir_path = 'export/multivariate_metrics/'
    if not os.path.isdir(dir_path):  # test if directory exist
        os.makedirs(dir_path)

    nf_a_id = {'pv': 10,
               'wind': 8,
               'load': 1}
    nf_umnn_id = {'pv': 3,
               'wind': 1,
               'load': 1}
    # ------------------------------------------------------------------------------------------------------------------
    # GEFcom IJF_paper case study
    # Solar track: 3 zones
    # Wind track: 10 zones
    # Load track: 1 zones
    # 50 days picked randomly per zone for the VS and TEST sets
    # ------------------------------------------------------------------------------------------------------------------
    es_res = []
    vs_res = []
    for tag in ['wind', 'pv', 'load']:
        if tag == 'pv':
            # WARNING: the time periods where PV is always 0 (night hours) are removed -> there are 8 periods removed
            # The index of the time periods removed are provided into indices
            data, indices = pv_data(path_name='../data/solar_new.csv', test_size=50, random_state=0)
            nb_zones = 3

        elif tag == 'wind':
            data = wind_data(path_name='../data/wind_data_all_zone.csv', test_size=50, random_state=0)
            nb_zones = 10
            indices = []

        elif tag == 'load':
            data = load_data(path_name='../data/load_data_track1.csv', test_size=50, random_state=0)
            nb_zones = 1
            indices = []

        df_x_LS = data[0].copy()
        df_y_LS = data[1].copy()
        df_x_VS = data[2].copy()
        df_y_VS = data[3].copy()
        df_x_TEST = data[4].copy()
        df_y_TEST = data[5].copy()

        nb_days_LS = int(len(df_y_LS) /nb_zones)
        nb_days_VS = int(len(df_y_VS) /nb_zones )
        nb_days_TEST = int(len(df_y_TEST) /nb_zones)

        print('#LS %s days #VS %s days # TEST %s days' % (nb_days_LS, nb_days_VS, nb_days_TEST))

        # ------------------------------------------------------------------------------------------------------------------
        # Rebuilt the PV observations with the removed time periods
        # ------------------------------------------------------------------------------------------------------------------
        non_null_indexes = list(np.delete(np.asarray([i for i in range(24)]), indices))

        if tag == 'pv':
            # Rebuilt the PV observations with the removed time periods
            df_y_TEST.columns = non_null_indexes
            for i in indices:
                df_y_TEST[i] = 0
            df_y_TEST = df_y_TEST.sort_index(axis=1)
        # observations shape = (n_days, 24)

        # scenarios shape = (24*n_days, n_s)
        s_umnn_TEST = read_file(dir='scenarios/nfs/', name='scenarios_' + tag + '_UMNN_M_' + str(nf_umnn_id[tag]) + '_0_100_TEST')
        # s_an_TEST = read_file(dir='scenarios/nfs/', name='scenarios_' + tag + '_AN_M_' + str(nf_a_id[tag]) + '_0_100_TEST')
        s_vae_TEST = read_file(dir='scenarios/vae/', name='scenarios_' + tag + '_VAElinear_1_0_100_TEST')
        s_gan_TEST = read_file(dir='scenarios/gan/', name='scenarios_' + tag + '_GAN_wasserstein_1_0_100_TEST')
        # s_gc_TEST = read_file(dir='scenarios/gc/', name='scenarios_' + tag + '_gc_100_TEST')
        s_rand_TEST = read_file(dir='scenarios/random/', name='scenarios_' + tag + '_random_100_TEST')

        # models = ['UMNN', 'AN', 'VAE', 'GAN', 'GC', 'RAND']
        models = ['NF', 'VAE', 'GAN', 'RAND']
        es_res_track = []
        vs_res_track = []
        for s, m in zip([s_umnn_TEST, s_vae_TEST, s_gan_TEST, s_rand_TEST], models):
            es_model = energy_score(s=s, y_true=df_y_TEST.values)
            print("%s %s ES: %.2f" %(tag, m, 100 * np.mean(es_model)))
            es_res_track.append(100 * np.mean(es_model))

            vs_model = variogram_score(s=s, y_true=df_y_TEST.values, beta=beta)
            print("%s %s VS: %.2f" %(tag, m, np.mean(vs_model)))
            vs_res_track.append(np.mean(vs_model))

        # DM test
        p_value_es_track = compute_DM(score_l=es_res_track, multivariate=False)
        p_value_vs_track = compute_DM(score_l=vs_res_track, multivariate=False)
        plot_DM(p_value=p_value_es_track, dir=dir_path, pdf_name=tag + '_ES_DM_test')
        plot_DM(p_value=p_value_vs_track, dir=dir_path, pdf_name=tag + '_VS_DM_test')

        es_res_track = np.asarray(es_res_track)
        vs_res_track = np.asarray(vs_res_track)
        es_res.append(es_res_track)
        vs_res.append(vs_res_track)
    es_res = np.asarray(es_res)
    vs_res = np.asarray(vs_res)