# -*- coding: UTF-8 -*-

""""
Classifier-based metric.
"""

import sys

from GEFcom2014 import read_file
from GEFcom2014.forecast_quality import build_data_eval_cond, build_true_data_cond, build_true_data, build_data_eval
from GEFcom2014.utils import dump_file

sys.path.insert(1, '../../')
import math
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_recall_curve,roc_curve, roc_auc_score, average_precision_score


def compute_prc_roc(true_data:list, model: str, tag: str, cfl_config: dict, pv_indices:np.array, i_clf:int=0, cond:bool=True):
    """
    Compute the prc and roc using a classifier.
    :param model:
    :param tag:
    :param cfl_config:
    :return:
    """
    if cond:
        X_LS, y_LS, X_TEST, y_TEST = build_data_eval_cond(true_data=true_data, model=model, tag=tag, pv_indices=pv_indices, i_clf=i_clf)
    else:
        X_LS, y_LS, X_TEST, y_TEST = build_data_eval(true_data=true_data, model=model, tag=tag, i_clf=i_clf)

    xprc, yprc, xroc, yroc, lprc, lroc = fit_clf_prc_roc(cfl_params=cfl_config, X_LS=X_LS, y_LS=y_LS, X_TEST=X_TEST, y_TEST=y_TEST)

    return xprc, yprc, xroc, yroc, lprc, lroc

def plot_roc(x: np.array, y: np.array, l: float, dir_path: str, name: str, i_clf:int=0):
    """
    Plot the acc vs n estimators.
    """
    FONTSIZE = 10

    plt.figure()
    plt.plot(x,y,label="TS (auc: {:0.3f})".format(l))
    plt.xlabel('False Positive Rate', fontsize=FONTSIZE)
    plt.ylabel('True Positive Rate', fontsize=FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.title(name+ '_roc_' +str(i_clf))
    plt.grid(True)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dir_path + name + '_roc_' +str(i_clf) + '.pdf')
    plt.show()

def plot_prc(x: np.array, y: np.array, l: float, dir_path: str, name: str, i_clf:int=0):
    """
    Plot the acc vs n estimators.
    """
    FONTSIZE = 10

    plt.figure()
    plt.plot(x,y,label="TS (auc: {:0.3f})".format(l))
    plt.xlabel('Recall', fontsize=FONTSIZE)
    plt.ylabel('Precision', fontsize=FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.title(name+ '_prc_'+str(i_clf))
    plt.grid(True)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dir_path + name + '_prc_' + '_'+str(i_clf) + '.pdf')
    plt.show()

def fit_clf_prc_roc(cfl_params, X_LS: list, y_LS: list, X_TEST: list, y_TEST: list):
    """
    Learn an ExtraTreesClassifier.
    :param cfl_params:
    :param X_LS:
    :param y_LS:
    :param X_TEST:
    :param y_TEST:
    :return:
    """

    n_estimators = cfl_params['n_estimators']

    clf = ExtraTreesClassifier(max_depth=cfl_params['max_depth'],
                                random_state=cfl_params['random_state'], 
                                n_jobs=-1,
                                n_estimators=n_estimators)

    clf.fit(X_LS[0],  y_LS[0])

    pred_TS = clf.predict_proba(X_TEST[0])

    precision_TS, recall_TS, _ = precision_recall_curve(y_TEST[0], pred_TS[:,1])

    fpr_TS, tpr_TS, _ = roc_curve(y_TEST[0], pred_TS[:,1])

    xprc = recall_TS
    yprc = precision_TS

    xroc = fpr_TS
    yroc = tpr_TS

    lprc = average_precision_score(y_TEST[0], pred_TS[:,1])
    lroc = roc_auc_score(y_TEST[0], pred_TS[:,1])

    return xprc, yprc, xroc, yroc, lprc, lroc

def plot_prc_multi_clf(res: list, dir_path: str, name: str, N_clf: int):
    """
    Plot several roc curves.
    :param res:
    :param dir_path:
    :param name:
    :param N_clf:
    :return:
    """

    prc_auc_ts = [res_i[2] for res_i in res]

    FONTSIZE = 10
    plt.figure()
    for res_i in res:
        plt.plot(res_i[0], res_i[1], 'tab:blue')
    plt.hlines(xmin=0, xmax=1, y=2, colors='tab:blue', label="TS auc mean: {:0.3f} std: {:0.3f}".format(np.mean(prc_auc_ts), np.std(prc_auc_ts)))
    plt.xlabel('Recall', fontsize=FONTSIZE)
    plt.ylabel('Precision', fontsize=FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.title(name + '_prc_' + str(N_clf))
    plt.grid(True)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dir_path + name + '_prc_' + str(N_clf) + '.pdf')
    plt.show()

def plot_roc_multi_clf(res: list, dir_path: str, name: str, N_clf: int):
    """
    Plot several roc curves.
    :param res:
    :param dir_path:
    :param name:
    :param N_clf:
    :return:
    """

    roc_auc_ts = [res_i[2] for res_i in res]

    FONTSIZE = 10
    plt.figure()
    for res_i in res:
        plt.plot(res_i[0], res_i[1], 'tab:blue')
    plt.hlines(xmin=0, xmax=1, y=2, colors='tab:blue', label="TS auc mean: {:0.3f} std: {:0.3f}".format(np.mean(roc_auc_ts), np.std(roc_auc_ts)))
    plt.xlabel('False Positive Rate', fontsize=FONTSIZE)
    plt.ylabel('True Positive Rate', fontsize=FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.title(name + '_roc_' + str(N_clf))
    plt.grid(True)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dir_path + name + '_roc_' + str(N_clf) + '.pdf')
    plt.show()


def plot_roc_all_model(res: dict, dir_path: str, N_clf: int, tag:str, models:list, labels:list):
    """
    Plot several roc curves.
    :param res:
    :param N_clf:
    :return:
    """
    colors = ['tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red']

    FONTSIZE = 15
    plt.figure()
    for m, c, l in zip(models, colors, labels):
        for res_i in res[m]:
            plt.plot(res_i[0], res_i[1], color=c)
        plt.hlines(xmin=0, xmax=1, y=2, colors=c, label=l)
    plt.xlabel('False Positive Rate', fontsize=FONTSIZE)
    plt.ylabel('True Positive Rate', fontsize=FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(True)
    legend = plt.legend(fontsize=1.5*FONTSIZE, ncol=2)
    legend.remove()
    plt.tight_layout()
    plt.savefig(dir_path + 'roc_' + tag +'_'+str(N_clf) + '.pdf')
    plt.show()

if __name__ == "__main__":

    print(os.getcwd())

    # --------------------------------------------------------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------------------------------------------------

    cond = True # True or False
    tag = 'pv'  # pv, wind, load
    model = 'NF-UMNN' # NF-UMNN, NF-A, VAE, GAN, GC, RAND
    name = tag + '_' + model
    models = ['NF-UMNN', 'NF-A', 'VAE', 'GAN', 'RAND']

    print(name)

    cfl_config = {
        "max_depth": None,
        "n_estimators": 1000,
        "random_state": None
    }

    if cond:
        true_data, pv_indices = build_true_data_cond(tag=tag)
    else:
        true_data, pv_indices = build_true_data(tag=tag)

    # --------------------------------------------------------------------------------------------------------------
    # 1. Reliability
    # Fit a single clf with a ratio False / true = n_s
    # --------------------------------------------------------------------------------------------------------------

    dir_path = 'export/clf_eval_'+str(cond)+'/reliability/'
    if not os.path.isdir(dir_path):  # test if directory exist
        os.makedirs(dir_path)

    # Compute the acc using a clf for a given model
    xprc, yprc, xroc, yroc, lprc, lroc = compute_prc_roc(true_data=true_data, model=model, tag=tag, pv_indices=pv_indices, cfl_config=cfl_config, cond=cond)
    plot_prc(xprc,yprc, lprc, dir_path=dir_path, name=name)
    plot_roc(xroc,yroc, lroc, dir_path=dir_path, name=name)

    # --------------------------------------------------------------------------------------------------------------
    # 2. Robustness
    # Fit N clf with a ratio False / true = 1 where each time new scenarios are picked in the LS, VS, and TEST sets
    # --------------------------------------------------------------------------------------------------------------

    dir_path = 'export/clf_eval_'+str(cond)+'/reliability/'
    if not os.path.isdir(dir_path):  # test if directory exist
        os.makedirs(dir_path)

    N_clf = 50
    res_final = dict()
    for model in models:
        name = tag + '_' + model
        print(name)
        res_roc = []
        res_prc = []
        for i_clf in range(N_clf):
            xprc, yprc, xroc, yroc, lprc, lroc = compute_prc_roc(true_data=true_data, model=model, tag=tag, pv_indices=pv_indices, cfl_config=cfl_config, i_clf=i_clf, cond=cond)
            print("{:s} iteration before end {:.0f}".format(model, N_clf-(i_clf+1)), end="\r", flush=True)
            # plot_prc(xprc, yprc, lprc, dir_path=dir_path, name=name, n_s=1, i_clf=i_clf)
            # plot_roc(xroc, yroc, lroc, dir_path=dir_path, name=name, n_s=1, i_clf=i_clf)
            res_prc.append([xprc, yprc, lprc])
            res_roc.append([xroc, yroc, lroc])

        #plot_roc_multi_clf(res=res_roc, dir_path=dir_path, name=name, N_clf=N_clf)
        #plot_prc_multi_clf(res=res_prc, dir_path=dir_path, name=name, N_clf=N_clf)
        res_final[model] = res_roc

    dump_file(dir=dir_path, name='res_final_roc_' + tag +'_'+str(N_clf), file=res_final)

    res_final = read_file(dir=dir_path, name='res_final_roc_' + tag +'_'+str(N_clf))
    plot_roc_all_model(res=res_final, dir_path=dir_path, N_clf=N_clf, tag=tag, models=['NF-UMNN', 'VAE', 'GAN'], labels=['NF', 'VAE', 'GAN'])

    for model in models:
        roc_auc = [res_i[2] for res_i in res_final[model]]
        print("%s AUC: %.3f" %(model, np.mean(roc_auc)))