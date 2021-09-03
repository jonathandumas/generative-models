# -*- coding: utf-8 -*-

import math
import os
import seaborn as sns
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def periods_where_pv_is_null(df_inputs:pd.DataFrame):
    """
    Compute the time periods where the PV generation is always 0 for the solar track.
    :param df_inputs: solar track data.
    :return: indices where PV is always 0.
    """
    # Determine time periods where PV generation is 0
    nb_days = int(df_inputs[df_inputs['ZONE_1'] == 1]['POWER'].shape[0] / 24)
    max_zone1 = df_inputs[df_inputs['ZONE_1'] == 1]['POWER'].values.reshape(nb_days, 24).max(axis=0)
    max_zone2 = df_inputs[df_inputs['ZONE_2'] == 1]['POWER'].values.reshape(nb_days, 24).max(axis=0)
    max_zone3 = df_inputs[df_inputs['ZONE_3'] == 1]['POWER'].values.reshape(nb_days, 24).max(axis=0)

    indices1 = np.where(max_zone1 == 0)[0]
    indices2 = np.where(max_zone2 == 0)[0]
    indices3 = np.where(max_zone3 == 0)[0]

    print('zone 1', indices1)
    print('zone 2', indices2)
    print('zone 3', indices3)

    return indices1

def wind_data(path_name: str, random_state: int = 0, test_size:int=2*12*2):
    """
    Build the wind power data for the GEFcom IJF_paper case study.
    """

    df_wind = pd.read_csv(path_name, parse_dates=True, index_col=0)
    ZONES = ['ZONE_' + str(i) for i in range(1, 10 + 1)]

    # INPUTS DESCRIPTION
    # The predictors included wind forecasts at two heights, 10 and 100 m above ground level, obtained from the European Centre for Medium-range Weather Forecasts (ECMWF).
    # These forecasts were for the zonal and meridional wind components (denoted u and v), i.e., projections of the wind vector on the west-east and south-north axes, respectively.

    # U10 zonal wind component at 10 m
    # V10 meridional wind component at 10 m
    # U100 zonal wind component at 100 m
    # V100 meridional wind component at 100 m

    # ------------------------------------------------------------------------------------------------------------------
    # Build derived features
    # cf winner GEFcom2014 wind track “Probabilistic gradient boosting machines for GEFCom2014 wind forecasting”
    # ------------------------------------------------------------------------------------------------------------------

    # the wind speed (ws), wind energy (we), and wind direction (wd) were as follows,
    # where u and v are the wind components provided and d is the density, for which we used a constant 1.0
    # ws = sqrt[u**2  + v**2]
    # we = 0.5 × d × ws**3
    # wd = 180/π × arctan(u, v)

    df_wind['ws10'] = np.sqrt(df_wind['U10'].values ** 2 + df_wind['V10'].values ** 2)
    df_wind['ws100'] = np.sqrt(df_wind['U100'].values ** 2 + df_wind['V100'].values ** 2)
    df_wind['we10'] = 0.5 * 1 * df_wind['ws10'].values ** 3
    df_wind['we100'] = 0.5 * 1 * df_wind['ws100'].values ** 3
    df_wind['wd10'] = np.arctan2(df_wind['U10'].values, df_wind['V10'].values) * 180 / np.pi
    df_wind['wd100'] = np.arctan2(df_wind['U100'].values, df_wind['V100'].values) * 180 / np.pi

    features = ['U10', 'V10', 'U100', 'V100', 'ws10', 'ws100', 'we10', 'we100', 'wd10', 'wd100']

    data_zone = []
    for zone in ZONES:
        df_var = df_wind[df_wind[zone] == 1].copy()
        nb_days = int(len(df_var) / 24)
        zones = [df_var[zone].values.reshape(nb_days, 24)[:, 0].reshape(nb_days, 1) for zone in ZONES]
        x = np.concatenate([df_var[col].values.reshape(nb_days, 24) for col in features] + zones, axis=1)
        y = df_var['TARGETVAR'].values.reshape(nb_days, 24)
        df_y = pd.DataFrame(data=y, index=df_var['TARGETVAR'].asfreq('D').index)
        df_x = pd.DataFrame(data=x, index=df_var['TARGETVAR'].asfreq('D').index)

        # Decomposition between LS, VS & TEST sets (TRAIN = LS + VS)
        df_x_train, df_x_TEST, df_y_train, df_y_TEST = train_test_split(df_x, df_y, test_size=test_size,random_state=random_state, shuffle=True)
        df_x_LS, df_x_VS, df_y_LS, df_y_VS = train_test_split(df_x_train, df_y_train, test_size=test_size,random_state=random_state, shuffle=True)

        data_zone.append([df_x_LS, df_y_LS, df_x_VS, df_y_VS, df_x_TEST, df_y_TEST])

        nb_days_LS = len(df_y_LS)
        nb_days_VS = len(df_y_VS)
        nb_days_TEST = len(df_y_TEST)
        print('#LS %s days #VS %s days # TEST %s days' % (nb_days_LS, nb_days_VS, nb_days_TEST))

    return [pd.concat([data_zone[i][j] for i in range(0, 9 + 1)], axis=0, join='inner') for j in range(0, 5 + 1)]


def load_data(path_name: str, random_state: int = 0, test_size:int=2*12*2):
    """
    Build the load power data for the GEFcom IJF_paper case study.
    """
    df_load = pd.read_csv(path_name, parse_dates=True, index_col=0)
    features = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10',
                'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w18', 'w19', 'w20',
                'w21', 'w22', 'w23', 'w24', 'w25']
    max_load = df_load['LOAD'].max()

    nb_days = int(len(df_load) / 24)
    x = np.concatenate([df_load[col].values.reshape(nb_days, 24) for col in features], axis=1)
    y = df_load['LOAD'].values.reshape(nb_days, 24) / max_load
    df_y = pd.DataFrame(data=y, index=df_load['LOAD'].asfreq('D').index)
    df_x = pd.DataFrame(data=x, index=df_load['LOAD'].asfreq('D').index)

    # Decomposition between LS, VS & TEST sets (TRAIN = LS + VS)
    df_x_train, df_x_TEST, df_y_train, df_y_TEST = train_test_split(df_x, df_y, test_size=test_size,
                                                                    random_state=random_state, shuffle=True)
    df_x_LS, df_x_VS, df_y_LS, df_y_VS = train_test_split(df_x_train, df_y_train, test_size=test_size,
                                                          random_state=random_state, shuffle=True)

    nb_days_LS = len(df_y_LS)
    nb_days_VS = len(df_y_VS)
    nb_days_TEST = len(df_y_TEST)
    print('#LS %s days #VS %s days # TEST %s days' % (nb_days_LS, nb_days_VS, nb_days_TEST))

    return df_x_LS, df_y_LS, df_x_VS, df_y_VS, df_x_TEST, df_y_TEST


def build_pv_features(df_var:pd.DataFrame, indices:np.array):
    """
    Build features for NFs multi-output.
    :param df_var: (n_periods, n_features)
    :param indices: index where PV generation is always 0.
    # INPUTS DESCRIPTION
    # Variable id. Variable name
    # 078.128 Total column liquid water (tclw)
    # 079.128 Total column ice water (tciw)
    # 134.128 Surface pressure (SP)
    # 157.128 Relative humidity at 1000 mbar (r)
    # 164.128 Total cloud cover (TCC)
    # 165.128 10-metre U wind component (10u)
    # 166.128 10-metre V wind component (10v)
    # 167.128 2-metre temperature (2T)
    # 169.128 Surface solar rad down (SSRD)
    # 175.128 Surface thermal rad down (STRD)
    # 178.128 Top net solar rad (TSR)
    # 228.128 Total precipitation (TP)
    """
    n_days = int(len(df_var) / 24)  # n days

    # Reshaping features from (24 * n_days,) to (n_days, 24) then drop time periods where PV is always 0
    y = df_var['POWER'].values.reshape(n_days, 24)
    y = np.delete(y, indices, axis=1)

    tclw = df_var['VAR78'].values.reshape(n_days, 24)
    tclw = np.delete(tclw, indices, axis=1)
    tciw = df_var['VAR79'].values.reshape(n_days, 24)
    tciw = np.delete(tciw, indices, axis=1)
    sp = df_var['VAR134'].values.reshape(n_days, 24)
    sp = np.delete(sp, indices, axis=1)
    rh = df_var['VAR157'].values.reshape(n_days, 24)
    rh = np.delete(rh, indices, axis=1)
    TCC = df_var['VAR164'].values.reshape(n_days, 24)
    TCC = np.delete(TCC, indices, axis=1)
    windU = df_var['VAR165'].values.reshape(n_days, 24)
    windU = np.delete(windU, indices, axis=1)
    windV = df_var['VAR166'].values.reshape(n_days, 24)
    windV = np.delete(windV, indices, axis=1)
    TT = df_var['VAR167'].values.reshape(n_days, 24)
    TT = np.delete(TT, indices, axis=1)
    SSRD = df_var['VAR169'].values.reshape(n_days, 24)
    SSRD = np.delete(SSRD, indices, axis=1)
    STRD = df_var['VAR175'].values.reshape(n_days, 24)
    STRD = np.delete(STRD, indices, axis=1)
    TSR = df_var['VAR178'].values.reshape(n_days, 24)
    TSR = np.delete(TSR, indices, axis=1)
    TP = df_var['VAR228'].values.reshape(n_days, 24)
    TP = np.delete(TP, indices, axis=1)
    zone1 = df_var['ZONE_1'].values.reshape(n_days, 24)[:, 0].reshape(n_days, 1)
    zone2 = df_var['ZONE_2'].values.reshape(n_days, 24)[:, 0].reshape(n_days, 1)
    zone3 = df_var['ZONE_3'].values.reshape(n_days, 24)[:, 0].reshape(n_days, 1)

    x = np.concatenate([TT, SSRD, np.multiply(SSRD, SSRD), np.multiply(SSRD, TT), rh, zone1, zone2, zone3], axis=1)

    return x,y

def pv_data(path_name: str, test_size:int, random_state:int=0):
    """
    Build the PV data for the GEFcom IJF_paper case study.
    """

    df_pv = pd.read_csv(path_name, parse_dates=True, index_col=0)

    ZONES = ['ZONE_1', 'ZONE_2', 'ZONE_3']
    indices = periods_where_pv_is_null(df_inputs=df_pv)

    data_zone = []
    for zone in ZONES:
        df_var = df_pv[df_pv[zone] == 1].copy()
        d_index = df_var['POWER'].asfreq('D').index
        x, y = build_pv_features(df_var=df_var, indices=indices)

        df_y = pd.DataFrame(data=y, index=d_index)
        df_x = pd.DataFrame(data=x, index=d_index)

        # Decomposition between LS, VS & TEST sets (TRAIN = LS + VS)
        df_x_train, df_x_TEST, df_y_train, df_y_TEST = train_test_split(df_x, df_y, test_size=test_size, random_state=random_state, shuffle=True)
        df_x_LS, df_x_VS, df_y_LS, df_y_VS = train_test_split(df_x_train, df_y_train, test_size=test_size, random_state=random_state, shuffle=True)

        data_zone.append([df_x_LS, df_y_LS, df_x_VS, df_y_VS, df_x_TEST, df_y_TEST])

        nb_days_LS = len(df_y_LS)
        nb_days_VS = len(df_y_VS)
        nb_days_TEST = len(df_y_TEST)
        print('%s #LS %s days #VS %s days # TEST %s days' % (zone, nb_days_LS, nb_days_VS, nb_days_TEST))

    return [pd.concat([data_zone[i][j] for i in [0, 1, 2]], axis=0, join='inner') for j in range(0, 5 + 1)], indices

def dump_file(dir:str, name: str, file):
    """
    Dump a file into a pickle.
    """
    file_name = open(dir + name + '.pickle', 'wb')
    pickle.dump(file, file_name)
    file_name.close()

def read_file(dir:str, name: str):
    """
    Read a file dumped into a pickle.
    """
    file_name = open(dir + name + '.pickle', 'rb')
    file = pickle.load(file_name)
    file_name.close()

    return file

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())

    # --------------------------------------------------------------------------------------------------------------
    # NEW DATASETS
    # --------------------------------------------------------------------------------------------------------------

    data, indices = pv_data(path_name='data/solar_new.csv', test_size=50, random_state=0)
    wind_data = wind_data(path_name='data/wind_data_all_zone.csv', test_size=50, random_state=0)
    load_data = load_data(path_name='data/load_data_track1.csv', test_size=50, random_state=0)