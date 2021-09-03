# -*- coding: utf-8 -*-

import os

import numpy as np

from GEFcom2014 import wind_data, load_data, pv_data
from GEFcom2014.forecast_quality import quantiles_and_evaluation
from GEFcom2014.utils import dump_file


def generate_random_scenarios(y: np.array, n_s: int, n_zones: int):
    """
    Sample random scenarios from observations.
    :param y: (size * n_zones, 24) with size = number of days for a given zone.
    """
    size = int(y.shape[0] / n_zones)
    scenarios = []
    # loop on all zones
    for z in range(n_zones):
        s_zone = []
        # loop on all days for a given zone
        for day in range(size):
            index_sampled = np.random.choice(size, n_s, replace=True)  # sample a list of indexes of length n_s
            s_day = y[z * size:(z + 1) * size][index_sampled].transpose()  # (24, n_s)
            s_zone.append(s_day)
        scenarios.append(np.concatenate(s_zone, axis=0))  # list of arrays (24 * TEST_size, n_s)
    return np.concatenate(scenarios, axis=0)  # (24 * size * n_zones, n_s)

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())

    tag = 'load' # pv, wind, load
    name = 'random'
    dir_path = 'export/'+name + '_'+tag+'/'
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

    s_LS = generate_random_scenarios(y=df_y_LS.values, n_s=n_s, n_zones=nb_zones)
    s_VS = generate_random_scenarios(y=df_y_VS.values, n_s=n_s, n_zones=nb_zones)
    s_TEST = generate_random_scenarios(y=df_y_TEST.values, n_s=n_s, n_zones=nb_zones)

    dump_file(dir=dir_path, name='scenarios_'+tag+'_'+name+'_' + str(n_s) + '_LS', file=s_LS)
    dump_file(dir=dir_path, name='scenarios_'+tag+'_'+name+'_' + str(n_s) + '_TEST', file=s_TEST)
    dump_file(dir=dir_path, name='scenarios_'+tag+'_'+name+'_' + str(n_s) + '_VS', file=s_VS)


    # ------------------------------------------------------------------------------------------------------------------
    # Build the PV quantiles from PV scenarios
    # ------------------------------------------------------------------------------------------------------------------
    N_q = 99
    quantiles_and_evaluation(dir_path=dir_path, s_VS=s_VS, s_TEST=s_TEST, N_q=N_q, df_y_VS=df_y_VS, df_y_TEST=df_y_TEST, name=name, ymax_plf=ymax_plf, ylim_crps=ylim_crps, tag=tag, nb_zones=nb_zones)