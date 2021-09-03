# -*- coding: utf-8 -*-

import math
import os
import pickle
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from GEFcom2014 import pv_data, wind_data, load_data, read_file

def build_true_data(tag: str):
    """
    Built the true data on LS, VS, and TEST sets
    :param tag: pv, load, wind.
    :return: true data into a list.
    """

    if tag == 'pv':
        # WARNING: the time periods where PV is always 0 (night hours) are removed -> there are 8 periods removed
        # The index of the time periods removed are provided into indices
        data, indices = pv_data(path_name='../data/solar_new.csv', test_size=50, random_state=0)

    elif tag == 'wind':
        data = wind_data(path_name='../data/wind_data_all_zone.csv', test_size=50, random_state=0)
        indices = []

    elif tag == 'load':
        data = load_data(path_name='../data/load_data_track1.csv', test_size=50, random_state=0)
        indices = []

    df_y_LS = data[1].copy()
    df_y_VS = data[3].copy()
    df_y_TEST = data[5].copy()

    non_null_indexes = list(np.delete(np.asarray([i for i in range(24)]), indices))

    if tag == 'pv':
        # Rebuilt the PV observations with the removed time periods
        df_y_LS.columns = non_null_indexes
        df_y_TEST.columns = non_null_indexes
        df_y_VS.columns = non_null_indexes

        for i in indices:
            df_y_LS[i] = 0
            df_y_VS[i] = 0
            df_y_TEST[i] = 0
        df_y_LS = df_y_LS.sort_index(axis=1)
        df_y_VS = df_y_VS.sort_index(axis=1)
        df_y_TEST = df_y_TEST.sort_index(axis=1)

    # Add only the zone one-hot encoding feature
    if tag == "pv":
        x_true_LS = np.hstack([df_y_LS.values, data[0].values[:,-3:].copy()])
        x_true_VS = np.hstack([df_y_VS.values, data[2].values[:,-3:].copy()])
        x_true_TEST = np.hstack([df_y_TEST.values, data[4].values[:,-3:].copy()])
    elif tag == "wind":
        x_true_LS = np.hstack([df_y_LS.values, data[0].values[:,-10:].copy()])
        x_true_VS = np.hstack([df_y_VS.values, data[2].values[:,-10:].copy()])
        x_true_TEST = np.hstack([df_y_TEST.values, data[4].values[:,-10:].copy()])
    else:
        # only one zone for the load track
        x_true_LS = df_y_LS.values
        x_true_VS = df_y_VS.values
        x_true_TEST = df_y_TEST.values

    # true value is class 1
    # false value is class 0
    y_true_LS = np.tile(1, (len(x_true_LS), 1))
    y_true_VS = np.tile(1, (len(x_true_VS), 1))
    y_true_TEST = np.tile(1, (len(x_true_TEST), 1))

    return [x_true_LS, x_true_VS, x_true_TEST, y_true_LS, y_true_VS, y_true_TEST], indices


def build_true_data_cond(tag: str):
    """
    Built the true data on LS, VS, and TEST sets
    :param tag: pv, load, wind.
    :return: true data into a list.
    """

    if tag == 'pv':
        # WARNING: the time periods where PV is always 0 (night hours) are removed -> there are 8 periods removed
        # The index of the time periods removed are provided into indices
        data, indices = pv_data(path_name='../data/solar_new.csv', test_size=50, random_state=0)

    elif tag == 'wind':
        data = wind_data(path_name='../data/wind_data_all_zone.csv', test_size=50, random_state=0)
        indices = []

    elif tag == 'load':
        data = load_data(path_name='../data/load_data_track1.csv', test_size=50, random_state=0)
        indices = []

    df_y_LS = data[1].copy()
    df_y_VS = data[3].copy()
    df_y_TEST = data[5].copy()

    non_null_indexes = list(np.delete(np.asarray([i for i in range(24)]), indices))

    # if tag == 'pv':
    #     # Rebuilt the PV observations with the removed time periods
    #     df_y_LS.columns = non_null_indexes
    #     df_y_TEST.columns = non_null_indexes
    #     df_y_VS.columns = non_null_indexes
    #
    #     for i in indices:
    #         df_y_LS[i] = 0
    #         df_y_VS[i] = 0
    #         df_y_TEST[i] = 0
    #     df_y_LS = df_y_LS.sort_index(axis=1)
    #     df_y_VS = df_y_VS.sort_index(axis=1)
    #     df_y_TEST = df_y_TEST.sort_index(axis=1)

    # Build the x and y of the clf
    # if tag == "pv":
    #     x_true_LS = np.hstack([df_y_LS.values, data[0].values[:,:-3].copy()])
    #     x_true_VS = np.hstack([df_y_VS.values, data[2].values[:,:-3].copy()])
    #     x_true_TEST = np.hstack([df_y_TEST.values, data[4].values[:,:-3].copy()])
    # elif tag == "wind":
    #     x_true_LS = np.hstack([df_y_LS.values, data[0].values[:,:-10].copy()])
    #     x_true_VS = np.hstack([df_y_VS.values, data[2].values[:,:-10].copy()])
    #     x_true_TEST = np.hstack([df_y_TEST.values, data[4].values[:,:-10].copy()])
    if tag == "pv":
        x_true_LS = np.hstack([df_y_LS.values, data[0].values[:,:].copy()])
        x_true_VS = np.hstack([df_y_VS.values, data[2].values[:,:].copy()])
        x_true_TEST = np.hstack([df_y_TEST.values, data[4].values[:,:].copy()])
    elif tag == "wind":
        x_true_LS = np.hstack([df_y_LS.values, data[0].values[:,:].copy()])
        x_true_VS = np.hstack([df_y_VS.values, data[2].values[:,:].copy()])
        x_true_TEST = np.hstack([df_y_TEST.values, data[4].values[:,:].copy()])
    else:
        x_true_LS = np.hstack([df_y_LS.values, data[0].values[:,:].copy()])
        x_true_VS = np.hstack([df_y_VS.values, data[2].values[:,:].copy()])
        x_true_TEST = np.hstack([df_y_TEST.values, data[4].values[:,:].copy()])

    # true value is class 1
    # false value is class 0
    y_true_LS = np.tile(1, (len(x_true_LS), 1))
    y_true_VS = np.tile(1, (len(x_true_VS), 1))
    y_true_TEST = np.tile(1, (len(x_true_TEST), 1))

    # Check shape of data
    # x_true shape should be LS/VS/TEST
    # PV: (720*3/50*3/50*3, 16 + 16*5 +3)
    # wind: (631*10/50*10/50*10, 24 + 24*10 +10)
    # load: (1999/50/50, 24 + 25*5)

    return [x_true_LS, x_true_VS, x_true_TEST, y_true_LS, y_true_VS, y_true_TEST], indices

def load_scenarios(dir: str, tag:str, name: str, i_clf:int=0):
    """
    Load the ith scenario per day on the LS and TEST sets.
    :param i_clf: ith scenario.
    """
    if tag == 'load':
        n_zones = 1
        ls_size = 1999 # days / zone
    elif tag == 'pv':
         n_zones = 3
         ls_size = 720  # days / zone

    elif tag == 'wind':
         n_zones = 10
         ls_size = 631 # days / zone

    # pick only one scenario per day
    # scenarios are into array of shape (24*n_days*_n_zone, n_s) with n_periods = 24*n_days*_n_zone
    s_model_temp = read_file(dir=dir, name='scenarios_' + tag + name+'_LS')
    n_days = int(s_model_temp.shape[0]/24)
    if len(s_model_temp) != int(24*ls_size*n_zones):
        print('WARNING with #LS')
        print(len(s_model_temp))
    s_model_ls = s_model_temp[:, i_clf].reshape(n_days, 24)  # (n_days*_n_zone, 24)

    s_model_temp = read_file(dir=dir, name='scenarios_' + tag + name+'_TEST')
    n_days = int(s_model_temp.shape[0]/24)
    if len(s_model_temp) != int(24*50*n_zones):
        print('WARNING with #TEST')
        print(len(s_model_temp))
    s_model_test = s_model_temp[:, i_clf].reshape(n_days, 24)  # (n_days*_n_zone, 24)

    # (n_days*_n_zone, 24)
    return s_model_ls, s_model_test


def build_data_eval(true_data:list, model: str, tag: str, i_clf:int=0):
    """
    Build the data for scenario evaluation using a classifier.
    :param model:
    :param tag:
    :return:
    """

    x_true_LS, x_true_VS, x_true_TEST, y_true_LS, y_true_VS, y_true_TEST = true_data

    # ------------------------------------------------------------------------------------------------------------------
    # 2. Load the scenarios
    # ------------------------------------------------------------------------------------------------------------------

    # load x_false -> scenarios
    # (n_days*_n_zone, 24)

    nf_a_id = {'pv': 10,
               'wind': 8,
               'load': 1}
    nf_umnn_id = {'pv': 3,
               'wind': 1,
               'load': 1}
    if model == 'NF-UMNN':
        # print(tag, nf_umnn_id[tag])
        x_false_LS, x_false_TEST = load_scenarios(dir='scenarios/nfs/', tag=tag, name='_UMNN_M_' + str(nf_umnn_id[tag]) + '_0_100', i_clf=i_clf)
    elif model == 'NF-A':
        # print(tag, nf_a_id[tag])
        x_false_LS, x_false_TEST = load_scenarios(dir='scenarios/nfs/', tag=tag, name='_AN_M_' + str(nf_a_id[tag]) + '_0_100', i_clf=i_clf)
    elif model == 'VAE':
        x_false_LS, x_false_TEST = load_scenarios(dir='scenarios/vae/', tag=tag, name='_VAElinear_1_0_100', i_clf=i_clf)
    elif model == 'GAN':
        x_false_LS, x_false_TEST = load_scenarios(dir='scenarios/gan/', tag=tag, name='_GAN_wasserstein_1_0_100', i_clf=i_clf)
    elif model == 'GC':
        x_false_LS, x_false_TEST = load_scenarios(dir='scenarios/gc/', tag=tag, name='_gc_100', i_clf=i_clf)
    elif model == 'RAND':
        x_false_LS, x_false_TEST = load_scenarios(dir='scenarios/random/', tag=tag, name='_random_100', i_clf=i_clf)

    if tag == 'pv':
        n_zones = 3
        # Add the zone one-hot encoding
        x_false_LS = np.hstack([x_false_LS, x_true_LS[:, -n_zones:]])
        x_false_TEST = np.hstack([x_false_TEST, x_true_TEST[:, -n_zones:]])
    elif tag == 'wind':
        n_zones = 10
        # Add the zone one-hot encoding
        x_false_LS = np.hstack([x_false_LS, x_true_LS[:, -n_zones:]])
        x_false_TEST = np.hstack([x_false_TEST, x_true_TEST[:, -n_zones:]])

    # 3. Build the dataset for the clf for a given time-series and model
    X_LS = np.concatenate((x_false_LS, x_true_LS), axis=0)
    X_TEST = np.concatenate((x_false_TEST, x_true_TEST), axis=0)

    # true value is class 1
    # false value is class 0
    y_false_LS = np.tile([0], (len(x_false_LS), 1))
    y_false_TEST = np.tile([0], (len(x_false_TEST), 1))

    y_LS = np.concatenate((y_false_LS, y_true_LS), axis=0)
    y_TEST = np.concatenate((y_false_TEST, y_true_TEST), axis=0)

    # 4. Build & fit the clf
    X_LS = [X_LS, x_false_LS, x_true_LS]
    X_TEST = [X_TEST, x_false_TEST, x_true_TEST]

    y_LS = [y_LS.reshape(-1), y_false_LS.reshape(-1), y_true_LS.reshape(-1)]
    y_TEST = [y_TEST.reshape(-1), y_false_TEST.reshape(-1), y_true_TEST.reshape(-1)]

    return X_LS, y_LS, X_TEST, y_TEST

def build_data_eval_cond(true_data:list, model: str, tag: str, pv_indices:np.array, i_clf:int=0):
    """
    Build the data for scenario evaluation using a classifier.
    :param model:
    :param tag:
    :return:
    """

    x_true_LS, x_true_VS, x_true_TEST, y_true_LS, y_true_VS, y_true_TEST = true_data

    # ------------------------------------------------------------------------------------------------------------------
    # 2. Load the scenarios
    # ------------------------------------------------------------------------------------------------------------------

    # load x_false -> scenarios
    # (n_days*_n_zone, 24)

    nf_a_id = {'pv': 10,
               'wind': 8,
               'load': 1}
    nf_umnn_id = {'pv': 3,
               'wind': 1,
               'load': 1}
    if model == 'NF-UMNN':
        # print(tag, nf_umnn_id[tag])
        x_false_LS, x_false_TEST = load_scenarios(dir='scenarios/nfs/', tag=tag, name='_UMNN_M_' + str(nf_umnn_id[tag]) + '_0_100', i_clf=i_clf)
    elif model == 'NF-A':
        # print(tag, nf_a_id[tag])
        x_false_LS, x_false_TEST = load_scenarios(dir='scenarios/nfs/', tag=tag, name='_AN_M_' + str(nf_a_id[tag]) + '_0_100', i_clf=i_clf)
    elif model == 'VAE':
        x_false_LS, x_false_TEST = load_scenarios(dir='scenarios/vae/', tag=tag, name='_VAElinear_1_0_100', i_clf=i_clf)
    elif model == 'GAN':
        x_false_LS, x_false_TEST = load_scenarios(dir='scenarios/gan/', tag=tag, name='_GAN_wasserstein_1_0_100', i_clf=i_clf)
    elif model == 'GC':
        x_false_LS, x_false_TEST = load_scenarios(dir='scenarios/gc/', tag=tag, name='_gc_100', i_clf=i_clf)
    elif model == 'RAND':
        x_false_LS, x_false_TEST = load_scenarios(dir='scenarios/random/', tag=tag, name='_random_100', i_clf=i_clf)

    if tag == 'pv':
        x_labels = ['$T$', '$I$', '$I^2$', '$I*T$', '$rh$']
        n_zones = 3
        # If PV dataset, remove the periods where the PV generation is always 0
        x_false_LS = np.delete(x_false_LS, pv_indices, axis=1)
        x_false_TEST = np.delete(x_false_TEST, pv_indices, axis=1)
    elif tag == 'wind':
        x_labels = ['$u^{10}$', '$v^{100}$', '$v^{10}$', '$v^{100}$', '$ws^{10}$', '$ws^{100}$', '$we^{10}$', '$we^{100}$', '$wd^{10}$', '$wd^{100}$']
        n_zones = 10
    elif tag == 'load':
        n_f = 25
        n_zones = 0
        x_labels = ['w_'+str(i) for i in range(1, n_f+1)]

    # print("Adding context...")
    if tag == "pv":
        tt = 24-8
    else:
        tt = 24
    len_feature = len(x_labels)*tt + n_zones

    # print("Adding {} features of {}".format(x_true_LS[:,-len_feature:].shape[1], tag))
    # Add the context to the scenarios (the context is in x_true); context = weather forecasts + zone one-hot encoding
    x_false_LS = np.hstack([x_false_LS, x_true_LS[:,-len_feature:]])
    x_false_TEST = np.hstack([x_false_TEST, x_true_TEST[:,-len_feature:]])
    # true value is class 1
    # false value is class 0
    y_false_LS = np.tile([0], (len(x_false_LS), 1))
    y_false_TEST = np.tile([0], (len(x_false_TEST), 1))

    # 3. Build the dataset for the clf for a given time-series and model
    X_LS = np.concatenate((x_false_LS, x_true_LS), axis=0)
    X_TEST = np.concatenate((x_false_TEST, x_true_TEST), axis=0)

    y_LS = np.concatenate((y_false_LS, y_true_LS), axis=0)
    y_TEST = np.concatenate((y_false_TEST, y_true_TEST), axis=0)

    X_LS = [X_LS, x_false_LS, x_true_LS]
    X_TEST = [X_TEST, x_false_TEST, x_true_TEST]

    y_LS = [y_LS.reshape(-1), y_false_LS.reshape(-1), y_true_LS.reshape(-1)]
    y_TEST = [y_TEST.reshape(-1), y_false_TEST.reshape(-1), y_true_TEST.reshape(-1)]

    # check shapes
    # X_LS, X_TEST should have shapes as follows:
    # PV:   (#LS *2 or #TEST *2, 16 + 16 * 5 +3)
    # wind: (#LS *2 or #TEST *2, 24 + 24 * 10 +10)
    # load: (#LS *2 or #TEST *2, 24 + 24 * 25)

    return X_LS, y_LS, X_TEST, y_TEST
if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
