# -*- coding: UTF-8 -*-

"""
DM-test for the CRPS and QS.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from GEFcom2014 import wind_data, load_data, pv_data, read_file
from GEFcom2014.forecast_quality import crps_per_period


def plf_per_period(quantiles: np.array, y_true: np.array):
    """
    Compute the averaged PLF per period over all quantiles for a given day.
    :param quantiles: (24, n_quantiles)
    :param y_true:  (24,)
    :return: PLF per period into an array (24, 1)
    """
    # quantile q from 0 to N_q -> add 1 to be from 1 to N_q into the PLF score
    n_periods = quantiles.shape[0]
    N_q = quantiles.shape[1]
    plf_p = []
    for p in range(n_periods):
        plf_p_q = 0
        for q in range(0, N_q):
            # for a given quantile compute the PLF over the entire dataset
            diff = (y_true[p] - quantiles[p, q]).item()
            # q from 0 to N_q-1 -> add 1 to be from 1 to N_q
            if diff > 0:
                plf_p_q += diff * ((q + 1) / (N_q + 1))
            elif diff < 0:
                plf_p_q += -diff * (1 - (q + 1) / (N_q + 1))
        plf_p.append(plf_p_q / N_q)
    return 100 * np.asarray(plf_p)


def plf_per_day(quantiles: np.array, y_true: np.array):
    """
    Compute the averaged PLF over all quantiles per time periods for all days.
    :param quantiles: (n_days*24, n_quantiles)
    :param y_true: (n_days, 24)
    :return: Averaged PLF over quantile (n_days, 24)
    """

    plf_day = []
    for d in range(y_true.shape[0]):
        plf_day.append(plf_per_period(quantiles=quantiles[24 * d:24 * (d + 1), :], y_true=y_true[d].reshape(24, 1)))
    return np.asarray(plf_day)  # (n_days, 24)

def DM_multivariate(score_1: np.array, score_2: np.array, multivariate:bool=False):
    """
    Function that performs the one-sided DM test.
    The test compares whether there is a difference in predictive accuracy between two scores 'score_1' and 'score_2' computed from two different forecasting model.
    Particularly, the one-sided DM test evaluates the null hypothesis H0 of the forecasting score_1 being larger (worse) than the forecasting errors 'score_2' vs the alternative hypothesis H1 of the errors of 'score_2' being smaller (better).
    Hence, rejecting H0 means that the forecast 'score_2' is significantly more accurate that forecast 'score_1'.
     Two versions of the test are possible:
        1. A univariate version for univariate score.
        2. A multivariate version for multivariate score.
    ----------
    The scores are computed over a testing set of n_days.
    Each day is composed of n_periods.
    Some scores such as the CRPS are univariate and are computed per marginal. Therefore, there is a CRPS per time period of the day.
    And the DM multivariate test should be performed.
    Some scores such as the ES or VS are multivariate and there is only a value of ES or VS per day and the DM univariate can be performed.
    :param :score_1 Array of shape (n_days,) if univariate version and  (n_days, n_periods) if multivariate version
    :param :score_2 Array of shape (n_days,) if univariate version and  (n_days, n_periods) if multivariate version
    :param :multivariate Version of the test

    :return the p-value after performing the test (a float).
    -------
    """

    # Checking that all time series have the same shape
    if score_1.shape != score_2.shape:
        raise ValueError('Both time series must have the same shape')

    # Computing the test statistic
    if multivariate:
        # Computing the loss differential series for the multivariate test
        loss_d = np.mean(np.abs(score_1), axis=1) - np.mean(np.abs(score_2), axis=1)  # shape = (n_days,)


    else:
        # Computing the loss differential series for the univariate test
        loss_d = score_1 - score_2  # shape = (n_days,)

    # Computing the loss differential size
    N = loss_d.size

    # Computing the test statistic
    mean_d = np.mean(loss_d)
    # print('mean_d %.4f' %(mean_d))
    var_d = np.var(loss_d, ddof=0)
    DM_stat = mean_d / np.sqrt((1 / N) * var_d)
    # print('DM_stat %.4f' %(DM_stat))

    p_value = 1 - stats.norm.cdf(DM_stat)

    return p_value


def compute_DM(score_l: list, multivariate:bool=False):
    """
    Compute the DM test for a list of scores (CRPS, PLF, etc).
    :param score_l: list of score of length L.
    :param multivariate: version of the test.
    :return: DM p-value into a np.array of shape (L,L)
    """

    p_value_tab = []
    for score1 in score_l:
        p_value_temp = []
        for score_2 in score_l:
            p_value = DM_multivariate(score_1=score1, score_2=score_2, multivariate=multivariate)
            # print('p-value %.2f' % (100 * p_value))
            p_value_temp.append(p_value)
        p_value_tab.append(p_value_temp)
    return np.asarray(p_value_tab)

if __name__ == "__main__":

    print(os.getcwd())

    dir_path = 'export/DM_test/'
    if not os.path.isdir(dir_path):  # test if directory exist
        os.makedirs(dir_path)

    score = 'plf'  # crps, plf
    multivariate = True

    nf_a_id = {'pv': 10,
               'wind': 8,
               'load': 1}
    nf_umnn_id = {'pv': 3,
               'wind': 1,
               'load': 1}

    for tag in ['pv', 'wind', 'load']:
    # tag = 'wind'  # pv, wind, load

        print('%s %s' %(tag, score))

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
            ylim_plf = [0, 4]
            ylim_crps = [0, 15]
            nb_zones = 3

        elif tag == 'wind':
            data = wind_data(path_name='../data/wind_data_all_zone.csv', test_size=50, random_state=0)
            ylim_plf = [0, 14]
            ylim_crps = [7, 20]
            nb_zones = 10
            indices = []

        elif tag == 'load':
            data = load_data(path_name='../data/load_data_track1.csv', test_size=50, random_state=0)
            ylim_plf = [0, 7]
            ylim_crps = [1, 11]
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

        # scenarios shape = (24*n_days, n_s)
        s_umnn_TEST = read_file(dir='scenarios/nfs/', name='scenarios_' + tag + '_UMNN_M_' + str(nf_umnn_id[tag]) + '_0_100_TEST')
        # s_an_TEST = read_file(dir='scenarios/nfs/', name='scenarios_' + tag + '_AN_M_'+str(nf_a_id[tag])+'_0_100_TEST')
        s_vae_TEST = read_file(dir='scenarios/vae/', name='scenarios_' + tag + '_VAElinear_1_0_100_TEST')
        s_gan_TEST = read_file(dir='scenarios/gan/', name='scenarios_' + tag + '_GAN_wasserstein_1_0_100_TEST')
        # s_gc_TEST = read_file(dir='scenarios/gc/', name='scenarios_' + tag +  '_gc_100_TEST')
        s_rand_TEST = read_file(dir='scenarios/random/', name='scenarios_' + tag + '_random_100_TEST')
        model_labels = ['NF', 'VAE', 'GAN', 'RAND']

        # --------------------------------------------------------------------------------------------------------------
        # PLF (n_days, n_periods)
        # --------------------------------------------------------------------------------------------------------------
        if score == 'plf':
            N_q = 99
            q_set = [i / (N_q + 1) for i in range(1, N_q + 1)]
            # Quantiles are generated into an array of shape (n_day*24, N_q), the same shape than scenarios
            q_umnn_TEST = np.quantile(s_umnn_TEST, q=q_set, axis=1).transpose()
            # q_an_TEST = np.quantile(s_an_TEST, q=q_set, axis=1).transpose()
            q_vae_TEST = np.quantile(s_vae_TEST, q=q_set, axis=1).transpose()
            q_gan_TEST = np.quantile(s_gan_TEST, q=q_set, axis=1).transpose()
            # q_gc_TEST = np.quantile(s_gc_TEST, q=q_set, axis=1).transpose()
            q_rand_TEST = np.quantile(s_rand_TEST, q=q_set, axis=1).transpose()

            plf_umnn = plf_per_day(quantiles=q_umnn_TEST, y_true=df_y_TEST.values)
            # plf_an = plf_per_day(quantiles=q_an_TEST, y_true=df_y_TEST.values)
            plf_vae = plf_per_day(quantiles=q_vae_TEST, y_true=df_y_TEST.values)
            plf_gan = plf_per_day(quantiles=q_gan_TEST, y_true=df_y_TEST.values)
            # plf_gc = plf_per_day(quantiles=q_gc_TEST, y_true=df_y_TEST.values)
            plf_rand = plf_per_day(quantiles=q_rand_TEST, y_true=df_y_TEST.values)
            score_l = [plf_umnn, plf_vae, plf_gan, plf_rand]
        elif score == 'crps':

            # --------------------------------------------------------------------------------------------------------------
            # CRPS (n_days, n_periods)
            # --------------------------------------------------------------------------------------------------------------
            max_s = 50
            crps_umnn, crps_umnn_d = crps_per_period(scenarios=s_umnn_TEST, y_true=df_y_TEST.values.reshape(-1), max_s=max_s)
            # crps_an, crps_an_d = crps_per_period_new(scenarios=s_an_TEST, y_true=df_y_TEST.values.reshape(-1),  max_s=max_s)
            crps_vae, crps_vae_d = crps_per_period(scenarios=s_vae_TEST, y_true=df_y_TEST.values.reshape(-1), max_s=max_s)
            crps_gan, crps_gan_d = crps_per_period(scenarios=s_gan_TEST, y_true=df_y_TEST.values.reshape(-1), max_s=max_s)
            # crps_gc, crps_gc_d = crps_per_period_new(scenarios=s_gc_TEST, y_true=df_y_TEST.values.reshape(-1),  max_s=max_s)
            crps_rand, crps_rand_d = crps_per_period(scenarios=s_rand_TEST, y_true=df_y_TEST.values.reshape(-1), max_s=max_s)

            score_l=[crps_umnn_d, crps_vae_d, crps_gan_d, crps_rand_d]

        # --------------------------------------------------------------------------------------------------------------
        # DM test with CRPS/PLF
        # --------------------------------------------------------------------------------------------------------------
        p_value = compute_DM(score_l=score_l, multivariate=multivariate)
        # --------------------------------------------------------------------------------------------------------------
        # PLOT DM test with CRPS
        # --------------------------------------------------------------------------------------------------------------

        import seaborn as sns

        pdf_name = tag + '_'+score+'_DM_test'
        FONTSIZE = 20
        plt.figure()
        sns.set(font_scale=1.5)
        sns_plot = sns.heatmap(100 * p_value, cmap='RdYlGn_r', fmt=".1f", linewidths=0.5, xticklabels=True, yticklabels=True, annot=False, vmin=0, vmax=10, annot_kws={"size": FONTSIZE})
        sns_plot.set_xticklabels(labels=model_labels, rotation='horizontal', fontsize=FONTSIZE)
        sns_plot.set_yticklabels(labels=model_labels, rotation=90, fontsize=FONTSIZE)
        sns_plot.figure.axes[-1].yaxis.label.set_size(FONTSIZE)
        plt.tight_layout()
        plt.savefig(dir_path + pdf_name + '.pdf')
        plt.show()
