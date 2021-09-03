# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

def scale_data_multi(x_LS: np.array, y_LS: np.array, x_VS: np.array, y_VS: np.array, x_TEST: np.array, y_TEST: np.array):
    """
    Scale data for NFs multi-output.
    """
    y_LS_scaler = StandardScaler()
    y_LS_scaler.fit(y_LS)
    y_LS_scaled = y_LS_scaler.transform(y_LS)
    y_VS_scaled = y_LS_scaler.transform(y_VS)
    y_TEST_scaled = y_LS_scaler.transform(y_TEST)

    x_LS_scaler = StandardScaler()
    x_LS_scaler.fit(x_LS)
    x_LS_scaled = x_LS_scaler.transform(x_LS)
    x_VS_scaled = x_LS_scaler.transform(x_VS)
    x_TEST_scaled = x_LS_scaler.transform(x_TEST)

    return x_LS_scaled, y_LS_scaled,  x_VS_scaled, y_VS_scaled, x_TEST_scaled, y_TEST_scaled, y_LS_scaler


def plot_loss(loss: np.array, nb_days: list, ylim: list, dir_path: str, name: str):
    """
    Plot the loss vs epoch.
    """
    FONTSIZE = 10
    nb_epoch = loss.shape[0]
    epoch_min = np.nanargmin(loss[:, 1])

    plt.figure()
    plt.plot(loss[:, 0], label='#LS ' + str(nb_days[0]))
    plt.plot(loss[:, 1], label='#VS ' + str(nb_days[1]))
    plt.plot(loss[:, 2], label='#TEST ' + str(nb_days[2]))
    plt.vlines(x=epoch_min, ymin=-100, ymax=100, colors='k',
               label='VS loss at ' + str(epoch_min) + ' = ' + str(round(loss[epoch_min, 1], 2)))
    plt.hlines(y=loss[epoch_min, 2], xmin=0, xmax=nb_epoch, colors='r',
               label='TEST loss at ' + str(epoch_min) + ' = ' + str(round(loss[epoch_min, 2], 2)))
    plt.xlabel('epoch', fontsize=FONTSIZE)
    plt.ylabel('ll loss', fontsize=FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlim(0, nb_epoch)
    plt.ylim(ylim[0], ylim[1])
    plt.title(name)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dir_path + name + '.pdf')
    plt.show()