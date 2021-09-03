# -*- coding: UTF-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from GEFcom2014 import pv_data, wind_data, load_data, read_file
from GEFcom2014.forecast_quality.utils_quality import plf_per_quantile, compute_reliability, crps_per_period

if __name__ == "__main__":

    """
    Quality scenario evaluation for all tracks (CRPS, QS, and reliability diagrams)
    """

    dir_path = 'export/all_tracks/'

    if not os.path.isdir(dir_path):  # test if directory exist
        os.makedirs(dir_path)

    # ------------------------------------------------------------------------------------------------------------------
    # GEFcom IJF_paper case study
    # Solar track: 3 zones
    # Wind track: 10 zones
    # Load track: 1 zones
    # 50 days picked randomly per zone for the VS and TEST sets
    # ------------------------------------------------------------------------------------------------------------------

    model_labels = ['NF', 'VAE', 'GAN']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    nb_scenarios = 100
    N_q = 99
    CRPS = True

    plf_all_models = dict()
    crps_all_models = dict()
    aq_all_models = dict()
    mae_r_all_models = dict()

    nf_a_id = {'pv': 10,
               'wind': 8,
               'load': 1}
    nf_umnn_id = {'pv': 3,
               'wind': 1,
               'load': 1}

    for tag in ['wind', 'pv', 'load']:
        # tag = 'wind'  # pv, wind, load

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
        # Quality metrics
        # --------------------------------------------------------------------------------------------------------------

        y_true = df_y_TEST.values.reshape(-1) # reshape from (#TEST, 24) to (24*,#TEST)
        # --------------------------------------------------------------------------------------------------------------
        # 0. Load scenarios on the TS for 'NF-UMNN', 'NF-A', 'VAE', 'GAN', and 'RAND'
        # --------------------------------------------------------------------------------------------------------------
        # scenarios shape = (24*n_days, n_s)
        s_umnn = read_file(dir='scenarios/nfs/', name='scenarios_' + tag + '_UMNN_M_' + str(nf_umnn_id[tag]) + '_0_100_TEST')
        # s_an = read_file(dir='scenarios/nfs/', name='scenarios_' + tag + '_AN_M_'+str(nf_a_id[tag])+'_0_100_TEST')
        s_vae = read_file(dir='scenarios/vae/', name='scenarios_' + tag + '_VAElinear_1_0_100_TEST')
        s_gan = read_file(dir='scenarios/gan/', name='scenarios_' + tag + '_GAN_wasserstein_1_0_100_TEST')
        # s_gc = read_file(dir='scenarios/gc/', name='scenarios_' + tag +  '_gc_100_TEST')
        s_rand = read_file(dir='scenarios/random/', name='scenarios_' + tag + '_random_100_TEST')
        scenarios_list = [s_umnn, s_vae, s_gan, s_rand]

        # PLot scenarios
        n_days = 1
        FONTSIZE = 15
        x_index = [i for i in range(1, n_days*24 + 1)]
        plt.figure()
        plt.plot(x_index, s_umnn[:n_days*24,:10], color='gray', linewidth=3, alpha=0.5)
        plt.ylim(0, 1)
        plt.tick_params(axis='both', labelsize=FONTSIZE)
        plt.tight_layout()
        plt.savefig(dir_path + tag + '_scenarios.pdf')
        plt.show()

        plt.figure()
        plt.plot(x_index, y_true[:n_days*24], color='red', linewidth=3)
        plt.ylim(0, 1)
        plt.tick_params(axis='both', labelsize=FONTSIZE)
        plt.tight_layout()
        plt.savefig(dir_path + tag + '_true.pdf')
        plt.show()

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

        print('%s PLF TS UMNN %.2f VAE %.2f GAN %.2f RAND %.2f' % (tag, plf_list[0].mean(), plf_list[1].mean(), plf_list[2].mean(), plf_list[3].mean()))
        print('')

        plf_all_models[tag] = plf_list[:-1]

        # --------------------------------------------------------------------------------------------------------------
        # 3. Reliability diagram
        # --------------------------------------------------------------------------------------------------------------
        aq_list = []
        for q in quantiles_list:
            aq_list.append(compute_reliability(y_true=y_true, y_quantile=q, tag=tag))

        aq_all_models[tag] = aq_list[:-1]

        mae_list = []
        for a in aq_list:
            mae_list.append(mean_absolute_error(y_true=np.array(q_set) * 100, y_pred=a))

        print('%s MAE TS UMNN %.2f VAE %.2f GAN %.2f RAND %.2f' % (tag, mae_list[0], mae_list[1], mae_list[2], mae_list[3]))
        print('')

        mae_r_all_models[tag] = mae_list[:-1]

        # --------------------------------------------------------------------------------------------------------------
        # CRPS
        # --------------------------------------------------------------------------------------------------------------
        if CRPS:
            max_s = 100
            crps_list = []
            for s in scenarios_list:
                crps, crps_d = crps_per_period(scenarios=s, y_true=y_true, max_s=max_s)
                crps_list.append(crps)
            print('%s CRPS TEST UMNN %.2f VAE %.2f GAN %.2f RAND %.2f' % (tag, 100 * crps_list[0].mean(), 100 * crps_list[1].mean(), 100 * crps_list[2].mean(), 100 * crps_list[3].mean()))
            print('')

            crps_all_models[tag] = crps_list[:-1]

    # --------------------------------------------------------------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------------------------------------------------------------
    """
    Plot the quantile score (PLF = Pinball Loss Function) per quantile on the TEST set of multiple generative models for all tracks.
    :param plf: list of the plf_score of multiple generative models. Each element of the list is an array.
    """

    x_index = [q for q in range(1, N_q + 1)]
    FONTSIZE = 15
    plt.figure(figsize=(5, 4))

    for l, c in zip(plf_all_models['wind'], colors):
         plt.plot(x_index, l, color=c, marker='P', linewidth=2)

    for l, c in zip(plf_all_models['load'], colors):
         plt.plot(x_index, l, color=c, linestyle="dashed", linewidth=2)

    for l, c, lab in zip(plf_all_models['pv'], colors, model_labels):
         plt.plot(x_index, l, color=c, label=lab, linewidth=2)

    plt.plot(x_index, [-20] * len(x_index), color='k', marker='P', label='wind')
    plt.plot(x_index, [-20] * len(x_index), color='k', label='PV')
    plt.plot(x_index, [-20] * len(x_index), color='k', linestyle="dashed", label='load')

    plt.vlines(x=(N_q  + 1) / 2, colors='k', ymin=0, ymax=7)
    plt.ylim(0, 7)
    plt.xlim(0, N_q  + 1)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xticks(ticks=[0, 20, 40, 60, 80, 100])
    plt.yticks(ticks=[0, 2, 4, 6])
    plt.xlabel('$q$', fontsize=FONTSIZE)
    plt.ylabel('%', fontsize=FONTSIZE)
    legend = plt.legend(fontsize=1.5*FONTSIZE, ncol=2)
    legend.remove()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_path + 'plf.pdf')
    plt.show()


    """
    Plot the CRPS on the TEST set of multiple generative models for all tracks.
    :param crps: list of the crps scores of multiple generative models. Each element of the list is an array.
    """
    if CRPS:
        FONTSIZE = 15
        plt.figure(figsize=(5, 4))

        for l, c in zip(crps_all_models['wind'], colors):
             plt.plot(100* l, color=c, marker='P', linewidth=2)

        for l, c in zip(crps_all_models['load'], colors):
             plt.plot(100* l, color=c, linestyle="dashed", linewidth=2)

        for l, c, lab in zip(crps_all_models['pv'], colors, model_labels):
             plt.plot(100* l, color=c, label=lab, linewidth=2)

        plt.tick_params(axis='both', labelsize=FONTSIZE)
        plt.xlabel('Hour', fontsize=FONTSIZE)
        plt.ylabel('%', fontsize=FONTSIZE)
        plt.xticks([0, 6, 12, 18, 23], ['1', '6', '12', '18', '24'])
        plt.yticks(ticks=[0, 2, 4, 6, 8, 10])
        legend = plt.legend(fontsize=1.5 * FONTSIZE, ncol=2)
        legend.remove()
        plt.grid(True)
        plt.ylim(0, 12)
        plt.xlim(0, 23)
        plt.tight_layout()
        plt.savefig(dir_path + 'crps.pdf')
        plt.show()



    """
    Plot the Reliablity diagram per quantile on the TEST set of multiple generative models for all tracks.
    :param aq: list of the aq scores of multiple generative models. Each element of the list is an array of shape (n_q,).
    """

    FONTSIZE = 15
    plt.figure(figsize=(5, 4))
    plt.plot(x_index, x_index, 'k', linewidth=2)

    for l, c in zip(aq_all_models['wind'], colors):
         plt.plot(x_index, l, color=c, marker='P')

    for l, c in zip(aq_all_models['load'], colors):
         plt.plot(x_index, l, color=c, linestyle="dashed")

    for l, c, lab in zip(aq_all_models['pv'], colors, model_labels):
         plt.plot(x_index, l, color=c, label=lab)

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('$q$', fontsize=FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    # plt.xticks(ticks=[i for i in range(0, 100 + 10, 10)])
    # plt.yticks(ticks=[i for i in range(0, 100 + 10, 10)])
    plt.xticks(ticks=[0, 20, 40, 60, 80, 100])
    plt.yticks(ticks=[0, 20, 40, 60, 80, 100])
    plt.ylabel('%', fontsize=FONTSIZE)
    legend = plt.legend(fontsize=1.5*FONTSIZE, ncol=2)
    legend.remove()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_path + 'reliability.pdf')
    plt.show()

    """
    Export only the legend
    """

    def export_legend(legend, filename="legend.pdf"):
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(dir_path + filename, dpi="figure", bbox_inches=bbox)

    x_index = [q for q in range(1, N_q + 1)]
    FONTSIZE = 15
    fig, ax = plt.subplots(1, figsize=(20,20))


    plt.plot(x_index, [-20] * len(x_index), color='tab:blue', label='NF', linewidth=2)
    plt.plot(x_index, [-20] * len(x_index), color='k', marker='P', label='wind')
    plt.plot(x_index, [-20] * len(x_index), color='tab:orange', label='VAE', linewidth=2)
    plt.plot(x_index, [-20] * len(x_index), color='k', label='PV')
    plt.plot(x_index, [-20] * len(x_index), color='tab:green', label='GAN', linewidth=2)
    plt.plot(x_index, [-20] * len(x_index), color='k', linestyle="dashed", label='load')
    plt.yticks(ticks=[-20] * len(x_index), labels=" ")
    plt.xticks(ticks=x_index, labels=" ")
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    legend = plt.legend(fontsize=1.5*FONTSIZE, ncol=3)
    export_legend(legend)
    plt.tight_layout()
    plt.show()

    x_index = [i for i in range(1, 24+1)]
    FONTSIZE = 15
    fig, ax = plt.subplots(1, figsize=(20,20))

    plt.plot(x_index, [-20] * len(x_index), color='gray', label='scenarios')
    plt.plot(x_index, [-20] * len(x_index), color='b', label='10 %')
    plt.plot(x_index, [-20] * len(x_index), color='k', label='50 %')
    plt.plot(x_index, [-20] * len(x_index), color='g', label='90 %')
    plt.plot(x_index, [-20] * len(x_index), color='r', label='obs')
    plt.yticks(ticks=[-20] * len(x_index), labels=" ")
    plt.xticks(ticks=x_index, labels=" ")
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    legend = plt.legend(fontsize=1.5*FONTSIZE, ncol=5)
    export_legend(legend, filename='legend_scenarios.pdf')
    plt.tight_layout()
    plt.show()