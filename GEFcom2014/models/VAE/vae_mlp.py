# -*- coding: UTF-8 -*-

import os
import json

import numpy as np
import torch

from GEFcom2014 import wind_data, load_data, pv_data
from GEFcom2014.models.VAE import VAElinear, fit_VAE, build_vae_scenarios
from GEFcom2014.forecast_quality import quantiles_and_evaluation
from GEFcom2014.utils import dump_file
from GEFcom2014.models import scale_data_multi, plot_loss
from torch.utils.benchmark import timer

from numpyencoder import NumpyEncoder


if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using gpu: %s ' % torch.cuda.is_available())

    # ------------------------------------------------------------------------------------------------------------------
    # GEFcom IJF_paper case study
    # Solar track: 3 zones
    # Wind track: 10 zones
    # Load track: 1 zones
    # 50 days picked randomly per zone for the VS and TEST sets
    #
    # A multi-output VAE:
    # Encoder = several MLP layers
    # Decoder = several MLP layers
    # ------------------------------------------------------------------------------------------------------------------

    tag = 'pv'  # pv, wind, load
    gpu = True  # put False to use CPU
    print('Using gpu: %s ' % torch.cuda.is_available())
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dir_path = 'export/vae_' + tag + '/'
    if not os.path.isdir(dir_path):  # test if directory exist
        os.makedirs(dir_path)

    # ------------------------------------------------------------------------------------------------------------------
    # Built the LS, VS, and TEST sets
    # ------------------------------------------------------------------------------------------------------------------

    if tag == 'pv':
        # WARNING: the time periods where PV is always 0 (night hours) are removed -> there are 8 periods removed
        # The index of the time periods removed are provided into indices
        data, indices = pv_data(path_name='../../data/solar_new.csv', test_size=50, random_state=0)
        ylim_loss = [0, 15]
        ymax_plf = 2.5
        ylim_crps = [0, 12]
        nb_zones = 3

    elif tag == 'wind':
        data = wind_data(path_name='../../data/wind_data_all_zone.csv', test_size=50, random_state=0)
        ylim_loss = [0, 30]
        ymax_plf = 8
        ylim_crps = [6, 12]
        nb_zones = 10
        indices = []

    elif tag == 'load':
        data = load_data(path_name='../../data/load_data_track1.csv', test_size=50, random_state=0)
        ylim_loss = [-5, 10]
        ymax_plf = 2
        ylim_crps = [0, 5]
        nb_zones = 1
        indices = []

    # reduce the LS size from 634 days to D days
    df_x_LS = data[0].copy()
    df_y_LS = data[1].copy()
    df_x_VS = data[2].copy()
    df_y_VS = data[3].copy()
    df_x_TEST = data[4].copy()
    df_y_TEST = data[5].copy()

    nb_days_LS = len(df_y_LS)
    nb_days_VS = len(df_y_VS)
    nb_days_TEST = len(df_y_TEST)
    print('#LS %s days #VS %s days # TEST %s days' % (nb_days_LS / nb_zones, nb_days_VS / nb_zones, nb_days_TEST / nb_zones))

    # ------------------------------------------------------------------------------------------------------------------
    # Scale the LS, VS, and TEST sets
    # ------------------------------------------------------------------------------------------------------------------

    # WARNING: use the scaler fitted on the TRAIN LS SET !!!!
    x_LS_scaled, y_LS_scaled, x_VS_scaled, y_VS_scaled, x_TEST_scaled, y_TEST_scaled, y_LS_scaler = scale_data_multi(x_LS=df_x_LS.values, y_LS=df_y_LS.values, x_VS=df_x_VS.values, y_VS=df_y_VS.values, x_TEST=df_x_TEST.values, y_TEST=df_y_TEST.values)

    non_null_indexes = list(np.delete(np.asarray([i for i in range(24)]), indices))

    if tag == 'pv':
        # Rebuilt the PV observations with the removed time periods
        df_y_TEST.columns = non_null_indexes
        for i in indices:
            df_y_TEST[i] = 0
        df_y_TEST = df_y_TEST.sort_index(axis=1)

        df_y_VS.columns = non_null_indexes
        for i in indices:
            df_y_VS[i] = 0
        df_y_VS = df_y_VS.sort_index(axis=1)
    # ------------------------------------------------------------------------------------------------------------------
    # Set the model
    # ------------------------------------------------------------------------------------------------------------------

    n_s = 100
    N_q = 99

    # Define the VAE hyper-parameters
    if tag == 'pv':
        nb_epoch = 300
        cf_VAE = {
            'name': 'VAElinear_1',
            'latent_s': 40,
            'enc_w': 200,
            'enc_l': 2,
            'dec_w': 200,
            'dec_l': 2,
            'weight_decay': 10 ** (-3.5),
            'learning_rate': 10 ** (-3.3),
        }

    elif tag == 'wind':
        nb_epoch = 200
        cf_VAE = {
            'name': 'VAElinear_1',
            'latent_s': 20,
            'enc_w': 200,
            'enc_l': 1,
            'dec_w': 200,
            'dec_l': 1,
            'weight_decay': 10 ** (-3.4),
            'learning_rate': 10 ** (-3.4),
        }

    elif tag == 'load':
        nb_epoch = 200
        cf_VAE = {
            'name': 'VAElinear_1',
            'latent_s': 5,
            'enc_w': 500,
            'enc_l': 1,
            'dec_w': 500,
            'dec_l': 1,
            'weight_decay': 10 ** (-4),
            'learning_rate': 10 ** (-3.9),
        }

    print('VAE latent_s %s enc_w %s enc_l %s dec_w %s dec_l %s weight_decay %.4e lr %.4e' % (cf_VAE['latent_s'], cf_VAE['enc_w'], cf_VAE['enc_l'], cf_VAE['dec_w'], cf_VAE['dec_l'], cf_VAE['weight_decay'], cf_VAE['learning_rate']))

    # Set the torch seed for result reproducibility
    torch_seed = 0
    torch.manual_seed(torch_seed)

    # Set the VAE configuration
    config = cf_VAE

    # --------------------------------------------------------------------------------------------------------------
    # Build the VAE
    # --------------------------------------------------------------------------------------------------------------
    name = tag + '_' + config['name'] + '_' + str(torch_seed)
    print(name)

    config['in_size'] = y_LS_scaled.shape[1]
    config['cond_in'] = x_LS_scaled.shape[1]

    # Dump into a json the VAE configuration
    with open(dir_path + config['name'] + '.json', 'w') as file:
        json.dump(config, file, cls=NumpyEncoder)

    VAE_model = VAElinear(latent_s=config['latent_s'], cond_in=config['cond_in'], in_size=config['in_size'], enc_w=config['enc_w'], enc_l=config['enc_l'], dec_w=config['dec_w'], dec_l=config['dec_l'], gpu=gpu)
    opt = torch.optim.Adam(VAE_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # --------------------------------------------------------------------------------------------------------------
    # Fit the VAE
    # --------------------------------------------------------------------------------------------------------------
    print('Fit VAE with %s epochs' % (nb_epoch))
    training_time = 0.
    start = timer()
    loss, best_vae, last_vae = fit_VAE(nb_epoch=nb_epoch, x_LS=x_LS_scaled, y_LS=y_LS_scaled, x_VS=x_VS_scaled,
                                        y_VS=y_VS_scaled, x_TEST=x_TEST_scaled, y_TEST=y_TEST_scaled,
                                        model=VAE_model,
                                        opt=opt, gpu=gpu)
    end = timer()
    training_time += end - start
    print('Training time %.2f s' %(training_time))
    epoch_min = np.nanargmin(loss[:, 1])
    print('epoch %s ll VS is min = %.2f ll TEST = %.2f' % (epoch_min, loss[epoch_min, 1], loss[epoch_min, 2]))

    # dump_file(dir=dir_path, name='loss_' + name, file=loss)
    # dump_file(dir=dir_path, name=name, file=best_vae)

    # --------------------------------------------------------------------------------------------------------------
    # Plot loss function
    # --------------------------------------------------------------------------------------------------------------
    plot_loss(loss=loss, nb_days=[nb_days_LS, nb_days_VS, nb_days_TEST], ylim=ylim_loss, dir_path=dir_path, name='ll_' + name)

    # --------------------------------------------------------------------------------------------------------------
    # Build scenarios on VS & TEST
    # --------------------------------------------------------------------------------------------------------------

    # best_flow = read_file(dir=dir_path, name=name)
    max_power = 1
    # Scenarios are generated into a dict of length nb days (#VS or # TEST sizes)
    # Each day of the dict is an array of shape (n_scenarios, 24)
    generation_time = 0.
    start = timer()
    # s_LS = build_vae_scenarios(n_s=n_s, x=x_LS_scaled, y_scaler=y_LS_scaler, model=best_vae, max=max_power, gpu=gpu, tag=tag, non_null_indexes=non_null_indexes)
    s_TEST = build_vae_scenarios(n_s=n_s, x=x_TEST_scaled, y_scaler=y_LS_scaler, model=best_vae, max=max_power, gpu=gpu, tag=tag, non_null_indexes=non_null_indexes)
    s_VS = build_vae_scenarios(n_s=n_s, x=x_VS_scaled, y_scaler=y_LS_scaler, model=best_vae, max=max_power, gpu=gpu, tag=tag, non_null_indexes=non_null_indexes)
    end = timer()
    generation_time += end - start
    print('Generation time (LS, VS, TEST) %.2f s' %(generation_time))
    # Export the scenarios
    # dict of nb_days with an array per day of shape = (nb_scenarios, 24)
    # dump_file(dir=dir_path, name='scenarios_' + name + '_' + str(n_s) + '_LS', file=s_LS)
    dump_file(dir=dir_path, name='scenarios_' + name + '_' + str(n_s) + '_TEST', file=s_TEST)
    dump_file(dir=dir_path, name='scenarios_' + name + '_' + str(n_s) + '_VS', file=s_VS)
    # scenarios_TEST = read_file(dir=dir_path, name='scenarios_' + name + '_' + str(nb_scenarios)+ '_TEST')
    # scenarios_VS = read_file(dir=dir_path, name='scenarios_' + name + '_' + str(nb_scenarios)+ '_VS')
    #
    # ------------------------------------------------------------------------------------------------------------------
    # Build the PV quantiles from PV scenarios
    # ------------------------------------------------------------------------------------------------------------------
    quantiles_and_evaluation(dir_path=dir_path, s_VS=s_VS, s_TEST=s_TEST, N_q=N_q, df_y_VS=df_y_VS, df_y_TEST=df_y_TEST, name=name, ymax_plf=ymax_plf, ylim_crps=ylim_crps, tag=tag, nb_zones=nb_zones)