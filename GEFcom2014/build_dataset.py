# -*- coding: UTF-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from GEFcom2014 import wind_data, load_data, read_file, pv_data
from GEFcom2014.utils import periods_where_pv_is_null

def differenciate(data:np.array):
    """
    Differentiating the four accumulated field: SSRD, STRD, TSR
    :return: differentiated data.
    """
    data_rolled = np.roll(data, 1, axis=1) # shift from one period
    data_rolled[:, 0] = 0
    data_diff = data - data_rolled
    data_diff[data_diff < 0] = 0
    return data_diff


def build_inputs(path_dir: str):
    """
    Build inputs.
    """
    df_inputs = pd.read_csv(path_dir, parse_dates=True, index_col=1)
    # Build an hot encoder to define the zone [1, 0, 1] = Zone 1
    df_inputs['ZONE_1'] = 0
    df_inputs['ZONE_2'] = 0
    df_inputs['ZONE_3'] = 0
    df_inputs.loc[df_inputs.ZONEID == 1, 'ZONE_1'] = 1
    df_inputs.loc[df_inputs.ZONEID == 2, 'ZONE_2'] = 1
    df_inputs.loc[df_inputs.ZONEID == 3, 'ZONE_3'] = 1
    df_inputs = df_inputs.drop('ZONEID', axis=1)

    return df_inputs

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())

    df_solar_data = build_inputs(path_dir="data/solar_track_predictors15.csv")
    indices = periods_where_pv_is_null(df_inputs=df_solar_data)
    solar_FEATURES = ['VAR78', 'VAR79', 'VAR134', 'VAR157', 'VAR164', 'VAR165', 'VAR166', 'VAR167', 'VAR169', 'VAR175', 'VAR178', 'VAR228', 'ZONE_1', 'ZONE_2', 'ZONE_3']
    solar_ZONES = ['ZONE_1', 'ZONE_2', 'ZONE_3']

    df_wind_data = pd.read_csv("data/wind_data_all_zone.csv", parse_dates=True, index_col=0)
    df_load_data= pd.read_csv("data/load_data_track1.csv", parse_dates=True, index_col=0)

    plt.figure()
    df_solar_data[df_solar_data['ZONE_1'] == 1]['POWER']['2012-04-01':'2012-04-03'].plot()
    plt.ylim(0, 1)
    plt.show()

    plt.figure()
    df_wind_data[df_wind_data['ZONE_1'] == 1]['TARGETVAR']['2012-01-01':'2012-01-20'].plot()
    plt.ylim(0, 1)
    plt.show()

    plt.figure()
    df_load_data['LOAD']['2005-01-01':'2005-01-20':].plot()
    plt.ylim(0, 250)
    plt.show()

    print('Solar data %s days' %(len(df_solar_data)/3/24))
    print('Wind data %s days' %(len(df_wind_data)/10/24))
    print('Load data %s days' %(len(df_load_data)/24))

    # --------------------------------------------------------------------------------------------------------------
    # PV dataset
    # --------------------------------------------------------------------------------------------------------------

    ssrd = df_solar_data['VAR169'].values.reshape(2463,24) / 3600 # from J/m2 to W/m2
    ssrd_diff = differenciate(data=ssrd)

    strd = df_solar_data['VAR175'].values.reshape(2463,24) / 3600 # from J/m2 to W/m2
    strd_diff = differenciate(data=strd)

    tsr = df_solar_data['VAR178'].values.reshape(2463,24) / 3600 # from J/m2 to W/m2
    tsr_diff = differenciate(data=tsr)

    tp = df_solar_data['VAR228'].values.reshape(2463,24)
    tp_diff = differenciate(data=tp)

    df_pv_new = df_solar_data.copy()
    df_pv_new['VAR169'] = ssrd_diff.reshape(-1)
    df_pv_new['VAR175'] = strd_diff.reshape(-1)
    df_pv_new['VAR178'] = tsr_diff.reshape(-1)
    df_pv_new['VAR228'] = tp_diff.reshape(-1)

    ZONES = ['ZONE_1', 'ZONE_2', 'ZONE_3']

    # Shift PV dataset from 10 periods for each zone and drop the first day
    df_list = []
    for zone in ZONES:
        df_pv_temp = df_pv_new[df_pv_new[zone] == 1].shift(periods=10)['2012-04-02 01':].copy()
        df_list.append(df_pv_temp)
    df_pv_final = pd.concat(df_list, axis=0, join='inner')

    plt.figure()
    df_pv_final[df_pv_final['ZONE_1'] == 1]['POWER']['2012-04-02':'2012-04-04'].plot()
    plt.ylim(0, 1)
    plt.show()

    df_pv_final.to_csv("data/solar_new.csv")

    # --------------------------------------------------------------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------------------------------------------------------------

    data, indices = pv_data(path_name='data/solar_new.csv', test_size=50, random_state=0)
    wind_data = wind_data(path_name='data/wind_data_all_zone.csv', test_size=50, random_state=0)
    load_data = load_data(path_name='data/load_data_track1.csv', test_size=50, random_state=0)

    dir_export = 'export_new/'
    FONTSIZE = 10
    plt.figure()
    plt.plot(wind_data[5].values.reshape(-1)[:3*24], 'r', linewidth=3,label='x')
    plt.ylim(-0.1, 1.1)
    plt.axis('off')
    plt.legend(fontsize=1.5*FONTSIZE)
    plt.tight_layout()
    # plt.savefig(dir_export + 'wind.pdf')
    plt.show()

    plt.figure()
    plt.plot(wind_data[4].values[:,:24].reshape(-1)[:3*24], 'b', linewidth=3, label='y')
    plt.ylim(-3.5, 8)
    plt.axis('off')
    plt.legend(fontsize=1.5*FONTSIZE)
    # plt.ylabel('m/s', fontsize=FONTSIZE)
    plt.tight_layout()
    # plt.savefig(dir_export + 'u10.pdf')
    plt.show()

    s_vae_TEST = read_file(dir='forecast_quality/scenarios/vae/', name='scenarios_wind_VAElinear_1_0_100_TEST')

    plt.figure()
    plt.plot(s_vae_TEST[:3*24,:8], 'gray', linewidth=3)
    plt.plot(s_vae_TEST[:3*24,9], 'gray', linewidth=3, label='$\hat{x}$')
    plt.ylim(-0.1, 1.1)
    plt.axis('off')
    plt.legend(fontsize=1.5*FONTSIZE)
    # plt.ylabel('%', fontsize=FONTSIZE)
    plt.tight_layout()
    # plt.savefig(dir_export + 'wind_s.pdf')
    plt.show()
