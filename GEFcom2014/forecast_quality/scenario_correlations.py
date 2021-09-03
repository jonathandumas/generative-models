# -*- coding: UTF-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GEFcom2014 import pv_data, wind_data, load_data, read_file
from GEFcom2014.forecast_quality import plot_corr_heat_map

if __name__ == "__main__":

    """
    Quality scenario evaluation: time-correlations between scenarios for a given context (weather forecasts).
    """

    tag = 'wind'  # pv, wind, load

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

    # --------------------------------------------------------------------------------------------------------------
    # Correlation matrices and autocorrelation plots
    # --------------------------------------------------------------------------------------------------------------

    dir_path = 'export/corr_matrices_'+ tag +'/'
    if not os.path.isdir(dir_path):  # test if directory exist
        os.makedirs(dir_path)

    # scenarios shape = (24*n_days, n_s)
    # select the zone
    zone = 1 # == 1 if load,  1 <= zone <= 3 for PV,   1 <= zone <= 10 for wind
    n_periods = nb_days_TEST * 24 # number of periods per zone

    s_umnn_TEST = read_file(dir='scenarios/nfs/', name='scenarios_' + tag + '_UMNN_M_' + str(nf_umnn_id[tag]) + '_0_100_TEST')[n_periods * (zone - 1):n_periods * zone, :]
    s_an_TEST = read_file(dir='scenarios/nfs/', name='scenarios_' + tag + '_AN_M_' + str(nf_a_id[tag]) + '_0_100_TEST')[n_periods * (zone - 1):n_periods * zone, :]
    s_vae_TEST = read_file(dir='scenarios/vae/', name='scenarios_' + tag + '_VAElinear_1_0_100_TEST')[n_periods * (zone - 1):n_periods * zone, :]
    s_gan_TEST = read_file(dir='scenarios/gan/', name='scenarios_' + tag + '_GAN_wasserstein_1_0_100_TEST')[n_periods * (zone - 1):n_periods * zone, :]
    s_gc_TEST = read_file(dir='scenarios/gc/', name='scenarios_' + tag + '_gc_100_TEST')[n_periods * (zone - 1):n_periods * zone, :]
    s_rand_TEST = read_file(dir='scenarios/random/', name='scenarios_' + tag + '_random_100_TEST')[n_periods * (zone - 1):n_periods * zone, :]
    model_labels = ['NF-UMNN', 'NF-A', 'VAE', 'GAN', 'GC', 'RAND', 'true']

    # --------------------------------------------------------------------------------------------------------------
    # y = PV, wind generation, load
    # x = context = weather forecasts
    # For a given x, several scenarios are generated: s_1|x, ...., s_n|x
    # Here only one scenario is selected for several context (days) of the TEST set
    # -> data with a matrix of shape (#TEST, 24) on which the correlations are computed
    # --------------------------------------------------------------------------------------------------------------
    vlim = [-0.5, 1]
    scenario_index = 0
    df_umnn_s_corr = pd.DataFrame(s_umnn_TEST[:,scenario_index].reshape(nb_days_TEST,24)).drop(columns=indices).corr()
    df_an_s_corr = pd.DataFrame(s_an_TEST[:,scenario_index].reshape(nb_days_TEST,24)).drop(columns=indices).corr()
    df_vae_s_corr = pd.DataFrame(s_vae_TEST[:,scenario_index].reshape(nb_days_TEST,24)).drop(columns=indices).corr()
    df_gan_s_corr = pd.DataFrame(s_gan_TEST[:,scenario_index].reshape(nb_days_TEST,24)).drop(columns=indices).corr()
    df_gc_s_corr = pd.DataFrame(s_gc_TEST[:,scenario_index].reshape(nb_days_TEST,24)).drop(columns=indices).corr()
    df_rand_s_corr = pd.DataFrame(s_rand_TEST[:,scenario_index].reshape(nb_days_TEST,24)).drop(columns=indices).corr()
    df_true_s_corr = pd.DataFrame(df_y_TEST.values[(zone-1)*nb_days_TEST:nb_days_TEST*zone]).drop(columns=indices).corr()

    for df_corr, label in zip([df_umnn_s_corr, df_an_s_corr, df_vae_s_corr, df_gan_s_corr, df_gc_s_corr, df_rand_s_corr, df_true_s_corr], model_labels):
        plot_corr_heat_map(dir=dir_path, name='s_' + str(scenario_index) + '_' + label, df_corr=df_corr, vlim=vlim)

    # --------------------------------------------------------------------------------------------------------------
    # y = PV, wind generation, load
    # x = context = conditionnement = weather forecasts
    # For a given x, several scenarios are generated: s_1|x, ...., s_n|x
    # Here the context is set for a given day and the correlations between scenarios are computed
    # -> data with a matrix of shape (n_s, 24) on which the correlations are computed
    # --------------------------------------------------------------------------------------------------------------
    vlim = [-0.5, 1]

    umnn_list = []
    an_list = []
    vae_list = []
    gan_list = []
    gc_list = []
    rand_list = []

    umnn_abs_list = []
    an_abs_list = []
    vae_abs_list = []
    gan_abs_list = []
    gc_abs_list = []
    rand_abs_list = []
    if tag == 'pv':
        # on day = 44 and 48 there are nan values when computing the correlation matrices for some of the models
        day_list = [i for i in range(0, 20+1)] + [i for i in range(22, 43+1)] + [45, 46, 47, 49]
    else:
        day_list = [i for i in range(0, 49+1)]

    for day in day_list:
        df_corr_umnn = pd.DataFrame(s_umnn_TEST[24*day:24*(day+1),:].transpose()).drop(columns=indices).corr()
        df_corr_an = pd.DataFrame(s_an_TEST[24*day:24*(day+1),:].transpose()).drop(columns=indices).corr()
        df_corr_vae = pd.DataFrame(s_vae_TEST[24*day:24*(day+1),:].transpose()).drop(columns=indices).corr()
        df_corr_gan = pd.DataFrame(s_gan_TEST[24*day:24*(day+1),:].transpose()).drop(columns=indices).corr()
        df_corr_gc = pd.DataFrame(s_gc_TEST[24*day:24*(day+1),:].transpose()).drop(columns=indices).corr()
        df_corr_rand = pd.DataFrame(s_rand_TEST[24*day:24*(day+1),:].transpose()).drop(columns=indices).corr()

        umnn_list.append(df_corr_umnn.values)
        an_list.append(df_corr_an.values)
        vae_list.append(df_corr_vae.values)
        gan_list.append(df_corr_gan.values)
        gc_list.append(df_corr_gc.values)
        rand_list.append(df_corr_rand.values)

        umnn_abs_list.append(df_corr_umnn.abs().values)
        an_abs_list.append(df_corr_an.abs().values)
        vae_abs_list.append(df_corr_vae.abs().values)
        gan_abs_list.append(df_corr_gan.abs().values)
        gc_abs_list.append(df_corr_gc.abs().values)
        rand_abs_list.append(df_corr_rand.abs().values)

        if day < 2:
            for df_corr, l in zip([df_corr_umnn, df_corr_an, df_corr_vae, df_corr_gan, df_corr_gc, df_corr_rand], model_labels):
                if tag == 'load' and (l == 'GAN'):
                    vlim = [0.9, 1]
                elif tag == 'load' and (l == 'VAE'):
                    vlim = [0.99, 1]
                else:
                    vlim = [-0.5, 1]
                plot_corr_heat_map(dir=dir_path, name='day_' + str(day) + '_' + l, df_corr=df_corr, vlim=vlim)
                plot_corr_heat_map(dir=dir_path, name='day_' + str(day) + '_' + l + '_abs', df_corr=df_corr.abs(), vlim=vlim)

            for s, l in zip([s_umnn_TEST, s_an_TEST, s_vae_TEST, s_gan_TEST, s_gc_TEST, s_rand_TEST], model_labels):
                # PLot scenarios
                FONTSIZE = 15
                x_index = [i for i in range(1, 24 + 1)]
                plt.figure()
                plt.plot(x_index, s[24 * day:24 * (day + 1), :50], color='gray', linewidth=3, alpha=0.5)
                plt.ylim(0, 1)
                plt.tick_params(axis='both', labelsize=FONTSIZE)
                plt.tight_layout()
                plt.savefig(dir_path + 'day_' + str(day) + '_' + l + '_scenarios.pdf')
                plt.show()

    # plot the average of the correlation matrices over the entire TEST set for all models
    for m,l in zip([umnn_list, an_list, vae_list, gan_list, gc_list, rand_list], model_labels):
        if tag == 'load' and (l == 'GAN'):
            vlim = [0.9, 1]
        elif tag == 'load' and (l == 'VAE'):
            vlim = [0.99, 1]
        else:
            vlim = [-0.5, 1]
        plot_corr_heat_map(dir=dir_path, name='average_' + l, df_corr=np.average(m, axis=0), vlim=vlim)

    # plot the ABS average of the correlation matrices over the entire TEST set for all models
    for m,l in zip([umnn_abs_list, an_abs_list, vae_abs_list, gan_abs_list, gc_abs_list, rand_abs_list], model_labels):
        if tag == 'load' and (l == 'GAN'):
            vlim = [0.9, 1]
        elif tag == 'load' and (l == 'VAE'):
            vlim = [0.99, 1]
        else:
            vlim = [-0.5, 1]
        plot_corr_heat_map(dir=dir_path, name='average_abs_' + l, df_corr=np.average(m, axis=0), vlim=vlim)