# -*- coding: utf-8 -*-

import math
import os
import torch
import random
import wandb

import pandas as pd
import numpy as np
from timeit import default_timer as timer

from sklearn.utils import shuffle


def build_nfs_scenarios(n_s: int, x: np.array, y_scaler, flow, conditioner_args, max:int=1, gpu:bool=True, tag:str= 'pv', non_null_indexes:list=[]):
    """
    Build scenarios for a NFs multi-output.
    Scenarios are generated into an array (n_periods, n_s) where n_periods = 24 * n_days
    :return: scenarios (n_periods, n_s)
    """
    # to assign the data to GPU with .to(device) on the data
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    flow.to(device)

    if tag == 'pv':
        n_periods_before = non_null_indexes[0]
        n_periods_after = 24 - non_null_indexes[-1] - 1
        print(n_periods_after, n_periods_before)

    n_days = len(x)
    nb_output, cond_in = conditioner_args['in_size'], conditioner_args['cond_in']
    time_tot = 0.
    scenarios = []
    for i in range(n_days):
        start = timer()
        # sample nb_scenarios per day
        predictions = flow.invert(z=torch.randn(n_s, nb_output).to(device), context=torch.tensor(np.tile(x[i, :], n_s).reshape(n_s, cond_in)).to(device).float()).cpu().detach().numpy()
        predictions = y_scaler.inverse_transform(predictions)

        # corrections -> genereration is always > 0 and < max capacity
        predictions[predictions < 0] = 0
        predictions[predictions > max] = max
        if tag == 'pv':
            # fill time period where PV is not 0 are given by non_null_indexes
            # for instance it could be [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            # then it is needed to add 0 for periods [0, 1, 2, 3] and [20, 21, 22, 23]

            scenarios_tmp = np.concatenate((np.zeros((predictions.shape[0], n_periods_before)), predictions, np.zeros((predictions.shape[0], n_periods_after))), axis=1)  # shape = (n_s, 24)
        else:
            scenarios_tmp = predictions
        scenarios.append(scenarios_tmp.transpose()) # list of arrays of shape (24, n_s)
        end = timer()
        time_tot += end - start
        print("day {:.0f} Approximate time left : {:2f} min".format(i, time_tot / (i + 1) * (n_days - (i + 1))/60), end="\r",flush=True)
        # if i % 20 == 0:
        #     print("day {:.0f} Approximate time left : {:2f} min".format(i, time_tot / (i + 1) * (nb_days - (i + 1)) / 60))
    print('Scenario generation time_tot %.1f min' % (time_tot / 60))
    return np.concatenate(scenarios,axis=0) # shape = (24*n_days, n_s)


def fit_NF(nb_epoch: int, x_LS: np.array, y_LS: np.array, x_VS: np.array, y_VS: np.array, x_TEST: np.array, y_TEST: np.array, flow, opt, batch_size:int=100, wdb:bool=False, gpu:bool=True):
    """
    Fit the NF.
    """
    # to assign the data to GPU with .to(device) on the data
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    flow.to(device)

    loss_list = []
    time_tot = 0.
    best_flow = flow
    # WARNING: batch size = 10 % #LS
    batch_size = int(0.1 * y_LS.shape[0])

    for epoch in range(nb_epoch):
        loss_tot = 0
        start = timer()

        # Shuffle the data randomly at each epoch
        seed = random.randint(0, 2000)
        x_LS_shuffled, y_LS_shuffled = shuffle(x_LS, y_LS, random_state=seed)

        # batch of 100 days
        # WARNING: if the NFs is single output, batch_size must be = 100 * 24 !!!
        i = 0
        loss_batch = 0
        batch_list = [i for i in range(batch_size, batch_size * y_LS.shape[0] // batch_size, batch_size)]
        for y_batch, x_batch in zip(np.split(y_LS_shuffled, batch_list), np.split(x_LS_shuffled, batch_list)):
            # We compute the log-likelihood as well as the base variable, check NormalizingFlow class for other possibilities
            ll, z = flow.compute_ll(x=torch.tensor(y_batch).to(device).float(), context=torch.tensor(x_batch).to(device).float())
            # Here we would like to maximize the log-likelihood of our model!
            loss = -ll.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_batch += loss.item()
            i += 1

        # LS loss is the average over all the batch
        loss_ls = loss_batch / i

        # VS loss
        ll_vs, z_vs = flow.compute_ll(x=torch.tensor(y_VS).to(device).float(), context=torch.tensor(x_VS).to(device).float())
        loss_vs = -ll_vs.mean().item()

        # TEST loss
        ll_test, z_test = flow.compute_ll(x=torch.tensor(y_TEST).to(device).float(), context=torch.tensor(x_TEST).to(device).float())
        loss_test = -ll_test.mean().item()

        # Save NF model when the VS loss is minimal
        loss_list.append([loss_ls, loss_vs, loss_test])

        ll_VS_min = np.nanmin(np.asarray(loss_list)[:, 1]) # ignore nan value when considering the min

        if not math.isnan(loss_vs) and loss_vs <= ll_VS_min:
            # print('NEW MIN ON VS at epoch %s loss_vs %.2f ll_VS_min %.2f' %(epoch, loss_vs, ll_VS_min))
            best_flow = flow # update the best flow
            # dump_file(dir=dir, name=name + '_'+str(epoch), file=best_flow)

        end = timer()
        time_tot += end - start

        if wdb:
            wandb.log({"ls loss": loss_ls})
            wandb.log({"vs loss": loss_vs})
            wandb.log({"test loss": loss_test})
            wandb.log({"vs min loss": ll_VS_min})

        if epoch % 10 == 0:
            # print("Epoch {:.0f} Approximate time left : {:2f} min - LS loss: {:4f} VS loss: {:4f} TEST loss: {:4f}".format(epoch, time_tot / (epoch + 1) * (nb_epoch - (epoch + 1)) / 60, loss_ls, loss_vs, loss_test))
            print("Epoch {:.0f} Approximate time left : {:2f} min - LS loss: {:4f} VS loss: {:4f} TEST loss: {:4f}".format(epoch, time_tot / (epoch + 1) * (nb_epoch - (epoch + 1)) / 60, loss_ls, loss_vs, loss_test), end="\r", flush=True)
    print('Fitting time_tot %.0f min' %(time_tot/60))
    return np.asarray(loss_list), best_flow, flow