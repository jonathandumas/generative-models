# -*- coding: utf-8 -*-

import math
import os
import pickle
import torch
import random
import wandb

import pandas as pd
import numpy as np
import torch.nn as nn

from torch.utils.benchmark import timer
from sklearn.utils import shuffle


class VAElinear(nn.Module):
    """
    VAE from A. Wehenkel with Linear layers.
    """
    def __init__(self, **kwargs):
        super(VAElinear, self).__init__()
        self.latent_s = kwargs['latent_s'] # Dim of the latent space
        self.in_size = kwargs['in_size']   # Dim of the random variable to model (PV, wind power, etc)
        self.cond_in = kwargs['cond_in']   # Dim of context (weather forecasts, etc)

        # Set GPU if available
        if kwargs['gpu']:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = 'cpu'

        # Build the Encoder & Decoder with Linear layers
        # x = target variable to generate samples (PV, wind, load, etc)
        # y = conditional variable (weather forecasts, etc)


        # The goal of the encoder is to model the posterior q(z|x,y)
        # q(z|x,y) is parametrized with an inference network NN_phi  that takes as input x,y
        # -> The output of the encoder are the mean and log of the covariance mu_z and log_sigma_z
        l_enc_net = [self.in_size + self.cond_in] + [kwargs['enc_w']] * kwargs['enc_l'] + [2 * self.latent_s]

        # The goal of the decoder is to model the likelihood p(x|z,y)
        # p(x|z,y) is parametrized with a generative network NN_theta that takes as input z
        # The output of the decoder is the approximation of p(x|z,y) with z sampled from N(g(x),h(x)) = N(mu_z, sigma_z)
        l_dec_net = [self.latent_s + self.cond_in] + [kwargs['dec_w']] * kwargs['dec_l'] + [self.in_size]

        # Build the Encoder
        self.enc_net = []
        for l1, l2 in zip(l_enc_net[:-1], l_enc_net[1:]):
            self.enc_net += [nn.Linear(l1, l2), nn.ReLU()]
        self.enc_net.pop()
        self.enc = nn.Sequential(*self.enc_net)

        # Build the Decoder
        self.dec_net = []
        for l1, l2 in zip(l_dec_net[:-1], l_dec_net[1:]):
            self.dec_net += [nn.Linear(l1, l2), nn.ReLU()]
        self.dec_net.pop()
        self.dec = nn.Sequential(*self.dec_net)

    def loss(self, x0, cond_in=None):
        """
        VAE loss function.
        Cf formulatio into the paper.
        :param x0: the random variable to fit.
        :param cond_in: context such as weather forecasts, etc.
        :return: the loss function = ELBO.
        """

        bs = x0.shape[0]
        if cond_in is None:
            cond_in = torch.empty(bs, self.cond_in)

        # Encoding -> NN_φ
        # The encoder ouputs mu_φ and log_sigma_φ
        #     μ(x,φ), log σ(x,φ)**2  = NN_φ(x)
        #     q(z∣x; φ)       = N(z; μ, σ^2*I)

        mu_phi, log_sigma_phi = torch.split(self.enc(torch.cat((x0, cond_in), 1)), self.latent_s, 1)
        # KL divergence
        # KL(q(z∣x; φ)∣∣p(z)) = 1/2 ∑ [1 + log(σ (x; φ)**2) − μ (x; φ)**2 − σ (x; φ)**2 ] because q(z∣x; φ) and p(z)follow a Normal distribution
        KL_phi = 0.5 * (1 + log_sigma_phi - mu_phi ** 2 - torch.exp(log_sigma_phi))
        KL_phi_new =  - KL_phi.sum(1).mean(0)

        # old KL_phi
        # KL_phi_old = (-log_sigma_phi + (mu_phi ** 2) / 2 + torch.exp(log_sigma_phi) / 2).sum(1).mean(0)

        # The reparameterization trick:
        z = mu_phi + torch.exp(log_sigma_phi) * torch.randn(mu_phi.shape, device=self.device)

        # Decoding -> NN_θ
        mu_x_pred = self.dec(torch.cat((z, cond_in), 1))

        # E_q(z∣x;ν) [log p(x∣z;θ)] ≃ ∑ log(p(x∣z;θ)) ≃ || [x - μ(x; θ)] / σ(x; θ) || ** 2 (MSE) because p(x∣z;θ) follows a Normal distribution
        KL_x = ((mu_x_pred.view(bs, -1) - x0) ** 2).view(bs, -1).sum(1).mean(0)

        loss = KL_x + KL_phi_new

        return loss

    def forward(self, x0):
        mu_z, log_sigma_z = torch.split(self.enc(x0.view(-1, *self.img_size)), self.latent_s, 1)
        mu_x_pred = self.dec(mu_z + torch.exp(log_sigma_z) * torch.randn(mu_z.shape, device=self.device))
        return mu_x_pred

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def sample(self, n_s=1, x_cond:np.array=None):
        """
        :param n_s: number of scenarios
        :param x_cond: context (weather forecasts, etc) into an array of shape (self.cond_in,)
        :return: samples into an array of shape (nb_samples, self.in_size)
        """
        # Generate samples from a multivariate Gaussian
        z = torch.randn(n_s, self.latent_s).to(self.device)

        context = torch.tensor(np.tile(x_cond, n_s).reshape(n_s, self.cond_in)).to(self.device).float()
        scenarios = self.dec(torch.cat((z, context), 1)).view(n_s, -1).cpu().detach().numpy()

        return scenarios


class VAEconv(nn.Module):
    """
    VAE with convolutional layers.
    """
    def __init__(self, **kwargs):
        super(VAEconv, self).__init__()
        self.latent_s = kwargs['latent_s'] # Dim of the latent space
        self.in_size = kwargs['in_size']   # Dim of the random variable to model (PV, wind power, etc)
        self.cond_in = kwargs['cond_in']   # Dim of context (weather forecasts, etc)

        # Set GPU if available
        if kwargs['gpu']:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = 'cpu'

        # Build the Encoder & Decoder with Linear layers
        l_enc_net = [self.in_size + self.cond_in] + [kwargs['enc_w']] * kwargs['enc_l'] + [2 * self.latent_s]
        l_dec_net = [self.latent_s + self.cond_in] + [kwargs['dec_w']] * kwargs['dec_l'] + [self.in_size]

        # Build the Encoder
        # self.enc_net = []
        # for l1, l2 in zip(l_enc_net[:-1], l_enc_net[1:]):
        #     self.enc_net += [nn.Linear(l1, l2), nn.ReLU()]
        # self.enc_net.pop()
        # self.enc = nn.Sequential(*self.enc_net)
        self.enc = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                              nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                              nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2, stride=1, padding=1), nn.ReLU(),
                              nn.Conv1d(in_channels=16, out_channels=8, kernel_size=2, stride=1, padding=1), nn.ReLU(),
                              nn.Flatten(start_dim=1, end_dim=-1),
                              nn.Linear(in_features=552, out_features=120), nn.ReLU(),
                              nn.Linear(in_features=120, out_features=60), nn.ReLU(),
                              nn.Linear(in_features=60, out_features=2 * self.latent_s),
                              )

        # FIXME: est ce que y a moyen de recuperer la dimension une fois le Flatten effectuee ??? car ici on passe de (bs, 8, 69) -> (bs, 552)

        # Build the Decoder
        # self.dec_net = []
        # for l1, l2 in zip(l_dec_net[:-1], l_dec_net[1:]):
        #     self.dec_net += [nn.Linear(l1, l2), nn.ReLU()]
        # self.dec_net.pop()
        # self.dec = nn.Sequential(*self.dec_net)
        #
        # FIXME specific to PV generation
        self.dec = nn.Sequential(nn.Linear(in_features=2 + (3 * 16 + 3), out_features=60), nn.ReLU(),
                              nn.Linear(in_features=60, out_features=120), nn.ReLU(),
                              nn.Linear(in_features=120, out_features=576), nn.ReLU(),
                              nn.Unflatten(dim=1, unflattened_size=torch.Size([36, 16])),
                              nn.Conv1d(in_channels=36, out_channels=18, kernel_size=2, stride=1, padding=0), nn.ReLU(),
                              nn.Conv1d(in_channels=18, out_channels=9, kernel_size=2, stride=1, padding=1), nn.ReLU(),
                              nn.Conv1d(in_channels=9, out_channels=18, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                              nn.Conv1d(in_channels=18, out_channels=1, kernel_size=3, stride=1, padding=1),
                              )


    def loss(self, x0, cond_in=None):
        """
        VAE loss function.
        We want to minimize the KL divergence between p(z|x) and its approximation q(z|x;v) where the variational parameters v index the family of distributions.
        KL(q(z∣x;ν)∣∣p(z∣x)) = E_q(z∣x;ν) [log {q(z∣x; ν) /  p(z∣x)}]
                             = E_q(z∣x;ν) [log q(z∣x; ν) − log p(x, z)] + log p(x)
        where ELBO(x; ν) is called the evidence lower bound objective and with
        ELBO(x; ν) = E_q(z∣x;ν) [log p(x, z) - log q(z∣x; ν)]
        Since log p(x) does not depend on ν, it can be considered as a constant, and minimizing the KL divergence is equivalent to maximizing the evidence lower bound, while being computationally tractable.

        ELBO(x; ν) = E_q(z;∣xν) [log p(x, z)    − log q(z∣x; ν)]
                   = E_q(z∣x;ν) [log p(x∣z)p(z) − log q(z∣x; ν)]
                   = E_q(z∣x;ν) [log p(x∣z)     − KL(q(z∣x; ν)∣∣p(z))]

        -> The likelihood p(x∣z; θ) is parameterized with a generative network NN_θ (or decoder) that takes as input and outputs parameters ϕ = NN_θ(z):
                μ(z,θ), σ(z,θ)  = NN_θ(z)
                p(x∣z; θ)       = N(x; μ, σ^2*I)
        -> The approximate posterior q(z∣x; φ) is parameterized with an inference network NN_φ (or encoder) that takes as input and outputs parameters ν = NN_φ(x):
                μ(x,φ), σ(x,φ)  = NN_φ(x)
                q(z∣x; φ)       = N(z; μ, σ^2*I)

        The KL divergence can be expressed analytically as:
        KL(q(z∣x; φ)∣∣p(z)) = 1/2 ∑ [1 + log(σ (x; φ)**2) − μ (x; φ)**2 − σ (x; φ)**2 ]

        E_q(z∣x;ν) [log p(x∣z)] can be estimated using a Monte Carlo method. It x is continous, p(x∣z) can be assumed to obey a Gaussian distribution, and the expression is the mean square error (MSE).
        E_q(z∣x;ν) [log p(x∣z)] ≃ ∑ log(p(x∣z)) ≃ || [x - μ(x; φ)] / σ(x; φ) || ** 2 (MSE)

        The reparameterization trick consists in re-expressing the variable z ∼ q(z∣x; φ)
        as some differentiable and invertible transformation of another random variable ϵ
        given x and φ: z = g(φ, x, ϵ), such that the distribution of ϵ is independent of x or φ.
        a common reparameterization is:
        p(ϵ) = N(ϵ; 0, I)
        z = μ(x; φ) + σ(x; φ) ⊙ ϵ

        :param x0: the random variable to fit.
        :param cond_in: context such as weather forecasts, etc.
        :return: the ELBO.
        """

        bs = x0.shape[0]
        if cond_in is None:
            cond_in = torch.empty(bs, self.cond_in)

        # Encoding
        # The encoder ouputs mu_φ and log_sigma_φ
        #     μ(x,φ), log σ(x,φ)**2  = NN_φ(x)
        #     q(z∣x; φ)       = N(z; μ, σ^2*I)

        # WARNING reshape because of the CNN from (bs, dim of enc_x) to (bs, 1, dim of enc_x)
        enc_x = torch.cat((x0, cond_in), dim=1)
        enc_x_reshaped = torch.reshape(enc_x, (enc_x.shape[0], 1, enc_x.shape[1]))
        mu_phi, log_sigma_phi = torch.split(self.enc(enc_x_reshaped), self.latent_s, 1)
        # KL divergence
        # KL(q(z∣x; φ)∣∣p(z)) = 1/2 ∑ [1 + log(σ (x; φ)**2) − μ (x; φ)**2 − σ (x; φ)**2 ]
        KL_phi = 0.5 * (1 + log_sigma_phi - mu_phi ** 2 - torch.exp(log_sigma_phi))
        KL_phi_new =  - KL_phi.sum(1).mean(0)

        # old KL_phi
        KL_phi_old = (-log_sigma_phi + (mu_phi ** 2) / 2 + torch.exp(log_sigma_phi) / 2).sum(1).mean(0)

        # The reparameterization trick:
        z = mu_phi + torch.exp(log_sigma_phi) * torch.randn(mu_phi.shape, device=self.device)

        # Decoding
        # WARNING reshape because of the CNN from (bs, 1, dim of enc_x) to (bs, dim of enc_x)
        mu_x_pred = self.dec(torch.cat((z, cond_in), 1))
        mu_x_pred_reshaped = torch.reshape(mu_x_pred, (mu_x_pred.shape[0], mu_x_pred.shape[2]))

        # E_q(z∣x;ν) [log p(x∣z)] ≃ ∑ log(p(x∣z)) ≃ || [x - μ(x; φ)] / σ(x; φ) || ** 2 (MSE)
        KL_x = ((mu_x_pred_reshaped.view(bs, -1) - x0) ** 2).view(bs, -1).sum(1).mean(0)

        loss = KL_x + KL_phi_new

        return loss

    def forward(self, x0):
        mu_z, log_sigma_z = torch.split(self.enc(x0.view(-1, *self.img_size)), self.latent_s, 1)
        mu_x_pred = self.dec(mu_z + torch.exp(log_sigma_z) * torch.randn(mu_z.shape, device=self.device))
        return mu_x_pred

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def sample(self, nb_samples=1, x_cond:np.array=None):
        """
        :param nb_samples: number of scenarios
        :param x_cond: context (weather forecasts, etc) into an array of shape (self.cond_in,)
        :return: samples into an array of shape (nb_samples, self.in_size)
        """
        # Generate samples from a multivariate Gaussian
        z = torch.randn(nb_samples, self.latent_s).to(self.device)

        context = torch.tensor(np.tile(x_cond, nb_samples).reshape(nb_samples, self.cond_in)).to(self.device).float()
        scenarios = self.dec(torch.cat((z, context), 1)).view(nb_samples, -1).cpu().detach().numpy()

        return scenarios

def fit_VAE(nb_epoch: int, x_LS: np.array, y_LS: np.array, x_VS: np.array, y_VS: np.array, x_TEST: np.array, y_TEST: np.array, model, opt, batch_size:int=100, wdb:bool=False, gpu:bool=True):
    """
    Fit the VAE.
    """
    # to assign the data to GPU with .to(device) on the data
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    model.to(device)

    # WARNING: batch size = 10 % #LS
    batch_size = int(0.1 * y_LS.shape[0])

    loss_list = []
    time_tot = 0.
    best_model = model
    for epoch in range(nb_epoch):
        start = timer()

        # Shuffle the data randomly at each epoch
        seed = random.randint(0, 2000)
        x_LS_shuffled, y_LS_shuffled = shuffle(x_LS, y_LS, random_state=seed)

        i = 0
        loss_batch = 0
        batch_list = [i for i in range(batch_size, batch_size * y_LS.shape[0] // batch_size, batch_size)]
        for y_batch, x_batch in zip(np.split(y_LS_shuffled, batch_list), np.split(x_LS_shuffled, batch_list)):
            loss = model.loss(x0=torch.tensor(y_batch).to(device).float(), cond_in=torch.tensor(x_batch).to(device).float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_batch += loss.item()
            i += 1

        # LS loss is the average over all the batch
        loss_ls = loss_batch / i

        # VS loss
        ll_vs = model.loss(x0=torch.tensor(y_VS).to(device).float(), cond_in=torch.tensor(x_VS).to(device).float())
        loss_vs = ll_vs.item()

        # TEST loss
        ll_test = model.loss(x0=torch.tensor(y_TEST).to(device).float(), cond_in=torch.tensor(x_TEST).to(device).float())
        loss_test = ll_test.item()

        # Save NF model when the VS loss is minimal
        loss_list.append([loss_ls, loss_vs, loss_test])

        # FIXME operation a faire en torch (mais pour l'instant la fonction torch.nanmin() n'existe pas ...
        # si operation possible en torch alors .item() sera appliqué à la fin du training pour éviter à chaque epoch de faire un transfert GPU vers CPU
        ll_VS_min = np.nanmin(np.asarray(loss_list)[:, 1]) # ignore nan value when considering the min

        if not math.isnan(loss_vs) and loss_vs <= ll_VS_min:
            # print('NEW MIN ON VS at epoch %s loss_vs %.2f ll_VS_min %.2f' %(epoch, loss_vs, ll_VS_min))
            best_model = model # update the best flow
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
    return np.asarray(loss_list), best_model, model

def build_vae_scenarios(n_s: int, x: np.array, y_scaler, model, max:int=1, gpu:bool=True, tag:str= 'pv', non_null_indexes:list=[]):
    """
    Build scenarios for a VAE multi-output.
    Scenarios are generated into an array (n_periods, n_s) where n_periods = 24 * n_days
    :return: scenarios (n_periods, n_s)
    """
    # to assign the data to GPU with .to(device) on the data
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    model.to(device)

    if tag == 'pv':
        n_periods_before = non_null_indexes[0]
        n_periods_after = 24 - non_null_indexes[-1] - 1
        print(n_periods_after, n_periods_before)

    nb_days = len(x)
    time_tot = 0.
    scenarios = []
    for i in range(nb_days):
        start = timer()
        # sample nb_scenarios per day of the VS or TEST sets
        predictions = model.sample(n_s=n_s, x_cond=x[i, :])
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
        print("day {:.0f} Approximate time left : {:2f} min".format(i, time_tot / (i + 1) * (nb_days - (i + 1))/60), end="\r",flush=True)
        # if i % 20 == 0:
        #     print("day {:.0f} Approximate time left : {:2f} min".format(i, time_tot / (i + 1) * (nb_days - (i + 1)) / 60))
    print('Scenario generation time_tot %.1f min' % (time_tot / 60))
    return np.concatenate(scenarios,axis=0) # shape = (24*n_days, n_s)


if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())

