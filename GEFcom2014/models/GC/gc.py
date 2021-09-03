# -*- coding: UTF-8 -*-

import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.distributions.empirical_distribution import ECDF

from scipy.interpolate import interp1d
from scipy.stats import norm

from GEFcom2014 import pv_data, wind_data, load_data
from GEFcom2014.forecast_quality import quantiles_and_evaluation
from GEFcom2014.utils import dump_file


def compute_ecdf(df_error: pd.DataFrame):
    """
    Compute the Estimated Cumulative Distribution Function (ECDF) for each time period of
    the error dataset of the random variable considered (PV, wind, etc).

    :param df_error: the error between the observations and point forecasts per time period.
                     shape = (n_samples, n_periods)
    :return: a dict (of length n_periods) with all the ECDF per time period.
    """

    ecdf = dict()
    for period in df_error.columns:
        ecdf[period] = ECDF(df_error[period])

    return ecdf


def compute_inv_ecdf(ecdf:dict, df_error:pd.DataFrame):
    """
    Compute the inverse of the Estimated Cumulative Distribution Function.

    :param ecdf: a dict (of length n_periods) with the ECDF per period.
    :return: a dict (of length n_periods) with the inverse of ECDF per period.
    """

    inv_ecdf = dict()
    for period in ecdf:
        slope_changes = sorted(set(df_error[period]))
        if len(slope_changes) > 1:
            sample_edf_values_at_slope_changes = [ecdf[period](item) for item in slope_changes]
            inv_ecdf[period] = interp1d(x=sample_edf_values_at_slope_changes, y=slope_changes, fill_value="extrapolate")

    return inv_ecdf

def plot_ecdf(dir:str, pdf_name: str, ecdf: dict, x_min: int, x_max: int):
    """
    Plot the ECDF of for over all the periods.
    """
    x = np.linspace(x_min, x_max, num=100)
    plt.figure(figsize=(10, 10))
    for key in ecdf.keys():
        plt.plot(x, ecdf[key](x), label=key)
    plt.legend(fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.title('ECDF', fontsize=FONTSIZE)
    plt.ylabel('Probability', fontsize=FONTSIZE)
    plt.xlabel('kW', fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dir + pdf_name + '_ecdf.pdf')
    plt.close('all')


def plot_inv_ecdf(dir:str, pdf_name: str, inv_ecdf: dict):
    """
    Plot the inverse of the ECDF over all the periods.
    """

    x = np.linspace(0.01, 1, num=1000)
    plt.figure(figsize=(10, 10))
    for market_period in inv_ecdf.keys():
        plt.plot(x, inv_ecdf[market_period](x), label=market_period)
    plt.legend(fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.title('Inverse of ECDF', fontsize=FONTSIZE)
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xlabel('Probability', fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dir + pdf_name + '_inv_ecdf.pdf')
    plt.close('all')


def gaussian_copula(df_corr:pd.DataFrame, nb_s:int, df_error:pd.DataFrame, inv_ecdf:dict):
    """
    1. Generate a sample g = (g_1, . . . , g_N) from a Normal distribution G = N(0, corr=df_corr).
    2. Transform each entry g_i through the standard normal cdf: u_i = phi(g_i) with phi = standard normal cdf = U[0,1]
    3. Apply to each entry u_i the inverse marginal cdf of Z_i = inv ECDF
    :param df_corr: correlation matrix of the Z that is the error dataset. It has been estimated previously.
    :param nb_s: number of scenarios to generate.
    :param df_error: the error dataset per market period.
        shape = (n_samples, N=n_period)
    :param inv_ecdf: the inverse ECDF of the error dataset per period.
    :return: df_s is a pd.DataFrame with the scenarios per period (n_s, n_periods)
    """

    # 1. Generate the sample G from a Normal distribution G = N(0, corr=df_corr).
    G_samples = np.random.multivariate_normal(mean=np.asarray([0] * df_corr.values.shape[0]), cov=df_corr.values, size=nb_s)

    # 2. Transform each entry g_i through the standard normal cdf: u_i = phi(g_i) with phi = standard normal cdf = U[0,1]
    U_samples = norm.cdf(G_samples)
    df_U = pd.DataFrame(data=U_samples, columns=df_error.columns)

    # 3. Apply to each entry u_i the inverse marginal cdf of Z_i = inv ECDF
    df_s = pd.DataFrame()
    for period in inv_ecdf:
        df_s[period] = inv_ecdf[period](df_U[period])

    return df_s


def build_gc_scenario(df_corr:pd.DataFrame, df_error:pd.DataFrame, inv_ecdf:dict, n_days: int, n_s: int, tag:str, max:float, non_null_indexes:list=[]):
    """
    Generate nb_s new scenarios per day of the LS, VS, and TEST sets
    :param n_days: #LS, #VS, or # TEST per zone
    :return: scenarios into a dict
    """

    if tag == 'pv':
        n_periods_before = non_null_indexes[0]
        n_periods_after = 24 - non_null_indexes[-1] - 1
        print(n_periods_after, n_periods_before)

    scenarios = []
    for d in range(n_days):
        df_s = gaussian_copula(df_corr=df_corr, nb_s=n_s, df_error=df_error, inv_ecdf=inv_ecdf) # (n_s, n_periods)
        df_s[df_s < 0] = 0
        df_s[df_s > max] = max
        predictions = df_s.values # (n_s, n_periods)
        if tag == 'pv':
            # fill time period where PV is not 0 are given by non_null_indexes
            # for instance it could be [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            # then it is needed to add 0 for periods [0, 1, 2, 3] and [20, 21, 22, 23]

            scenarios_tmp = np.concatenate((np.zeros((predictions.shape[0], n_periods_before)), predictions, np.zeros((predictions.shape[0], n_periods_after))), axis=1)  # shape = (n_s, 24)
        else:
            scenarios_tmp = predictions

        scenarios.append(scenarios_tmp.transpose())  # list of arrays of shape (24, n_s)

    return np.concatenate(scenarios, axis=0)

FONTSIZE = 10

if __name__ == "__main__":

    print(os.getcwd())

    # ----------------------------------------------------------------------------------------------------
    # GAUSSIAN COPULA (GC) scenario generation directly on the random variable
    # ----------------------------------------------------------------------------------------------------

    tag = 'pv' # pv, wind, load
    dir_path = 'export/gc_'+tag+'/'
    if not os.path.isdir(dir_path):  # test if directory exist
        os.makedirs(dir_path)

    n_s = 100
    if tag == 'pv':
        # WARNING: the time periods where PV is always 0 (night hours) are removed -> there are 8 periods removed
        # The index of the time periods removed are provided into indices
        data, indices = pv_data(path_name='../../data/solar_new.csv', test_size=50, random_state=0)
        ymax_plf = 4
        ylim_crps = [0, 14]
        nb_zones = 3

    elif tag == 'wind':
        data = wind_data(path_name='../../data/wind_data_all_zone.csv', test_size=50, random_state=0)
        ymax_plf = 14
        ylim_crps = [15, 18]
        nb_zones = 10
        indices = []

    elif tag == 'load':
        data = load_data(path_name='../../data/load_data_track1.csv', test_size=50, random_state=0)
        ymax_plf = 7
        ylim_crps = [5, 11]
        nb_zones = 1
        indices = []

    df_x_LS = data[0].copy()
    df_y_LS = data[1].copy()
    df_x_VS = data[2].copy()
    df_y_VS = data[3].copy()
    df_x_TEST = data[4].copy()
    df_y_TEST = data[5].copy()

    non_null_indexes = list(np.delete(np.asarray([i for i in range(24)]), indices))

    ls_size = int(len(df_y_LS)/nb_zones)
    vs_size = int(len(df_y_VS)/nb_zones)
    test_size = int(len(df_y_TEST)/nb_zones)

    print('#LS %s days #VS %s days # TEST %s days' % (ls_size, vs_size, test_size))
    ls_day_index = df_y_LS.index[:ls_size]
    s_LS_list = []
    s_VS_list = []
    s_TEST_list = []
    for z in range(nb_zones):
        # ----------------------------------------------------------------------------------------------------
        # 2. Compute the correlation matrix (pearson coefficient), the ECDF and its inverse

        df_y_LS_zone = pd.DataFrame(data=df_y_LS.values[z*ls_size:(z+1)*ls_size], index=ls_day_index)

        df_corr_LS = df_y_LS_zone.corr(method="pearson")
        ecdf_LS = compute_ecdf(df_error=df_y_LS_zone)
        inv_ecdf_LS = compute_inv_ecdf(ecdf=ecdf_LS, df_error=df_y_LS_zone)

        # Plot the correlation matrix with a heatmap
        # _r reverses the normal order of the color map 'RdYlGn'
        plt.figure(figsize=(6, 5))
        sns_plot = sns.heatmap(df_corr_LS, cmap='RdYlGn_r', fmt=".1f", linewidths=0.5, xticklabels=False, yticklabels=False,
                               annot=False, vmin=-.1, vmax=1)
        sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation='horizontal')
        plt.tight_layout()
        plt.savefig(dir_path + tag + '_corr.pdf')
        plt.show()

        # Plot the ECDF and inv ECDF
        plot_ecdf(dir=dir_path, pdf_name=tag, ecdf=ecdf_LS, x_min=0, x_max=1)
        plot_inv_ecdf(dir=dir_path, pdf_name=tag, inv_ecdf=inv_ecdf_LS)

        # -----------------------------------------------------------------------------------------------------------------
        # 3.1 Compute the error scenarios based on the estimated CDF and the estimated correlation matrix on the TRAIN set
        # Generate nb_s new scenarios per day of the LS, VS, and TEST sets
        max_cap = 1
        s_LS = build_gc_scenario(df_corr=df_corr_LS, df_error=df_y_LS_zone, inv_ecdf=inv_ecdf_LS, n_days=ls_size, n_s=n_s, tag=tag, max=max_cap, non_null_indexes=non_null_indexes)
        s_VS = build_gc_scenario(df_corr=df_corr_LS, df_error=df_y_LS_zone, inv_ecdf=inv_ecdf_LS, n_days=vs_size, n_s=n_s, tag=tag, max=max_cap, non_null_indexes=non_null_indexes)
        s_TEST = build_gc_scenario(df_corr=df_corr_LS, df_error=df_y_LS_zone, inv_ecdf=inv_ecdf_LS, n_days=test_size, n_s=n_s, tag=tag, max=max_cap, non_null_indexes=non_null_indexes)
        # dump_file(dir=dir_path, name='scenarios_' + tag + '_gc_' + str(n_s) + '_LS', file=s_LS)
        # dump_file(dir=dir_path, name='scenarios_' + tag + '_gc_' + str(n_s) + '_VS', file=s_VS)
        # dump_file(dir=dir_path, name='scenarios_' + tag + '_gc_' + str(n_s) + '_TEST', file=s_TEST)
        s_LS_list.append(s_LS)
        s_VS_list.append(s_VS)
        s_TEST_list.append(s_TEST)

    s_LS_all_zone = np.concatenate(s_LS_list, axis=0)
    s_VS_all_zone = np.concatenate(s_VS_list, axis=0)
    s_TEST_all_zone = np.concatenate(s_TEST_list, axis=0)

    dump_file(dir=dir_path, name='scenarios_' + tag + '_gc_' + str(n_s) + '_LS', file=s_LS_all_zone)
    dump_file(dir=dir_path, name='scenarios_' + tag + '_gc_' + str(n_s) + '_VS', file=s_VS_all_zone)
    dump_file(dir=dir_path, name='scenarios_' + tag + '_gc_' + str(n_s) + '_TEST', file=s_TEST_all_zone)

    # ------------------------------------------------------------------------------------------------------------------
    # Build the PV quantiles from PV scenarios
    # ------------------------------------------------------------------------------------------------------------------


    if tag == 'pv':
        # Rebuilt the PV observations with the removed time periods
        df_y_LS.columns = non_null_indexes
        for i in indices:
            df_y_LS[i] = 0
        df_y_LS = df_y_LS.sort_index(axis=1)

        df_y_TEST.columns = non_null_indexes
        for i in indices:
            df_y_TEST[i] = 0
        df_y_TEST = df_y_TEST.sort_index(axis=1)

        df_y_VS.columns = non_null_indexes
        for i in indices:
            df_y_VS[i] = 0
        df_y_VS = df_y_VS.sort_index(axis=1)


    N_q = 99
    name = 'gc'
    quantiles_and_evaluation(dir_path=dir_path, s_VS=s_VS_all_zone, s_TEST=s_TEST_all_zone, N_q=N_q, df_y_VS=df_y_VS, df_y_TEST=df_y_TEST, name=name, ymax_plf=ymax_plf, ylim_crps=ylim_crps, tag=tag, nb_zones=nb_zones)