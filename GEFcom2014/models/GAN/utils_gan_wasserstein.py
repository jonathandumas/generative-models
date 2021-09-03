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
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from torch.utils.benchmark import timer
from sklearn.utils import shuffle


class Generator_linear(nn.Module):
    """
    Define Generator class.
    Conditional generator using fully connected layers.
    """

    def __init__(self, **kwargs):

        """
        Generator constructor
        :param latent_s: Dim of the latent space
        :param cond_in: Dim of context (weather forecasts, etc)
        :param hidden_layer: number of hidden layers
        :param neurons_per_layer : number of neurons per hidden_layer
        :param in_size: Dim of the random variable to model (PV, wind power, etc)
        """

        super(Generator_linear, self).__init__()
        self.in_size = kwargs['in_size']   # Dim of the random variable to model (PV, wind power, etc)
        self.cond_in = kwargs['cond_in']   # Dim of context (weather forecasts, etc)
        self.latent_s = kwargs['latent_s'] # Dim of the latent space

        # Set GPU if available
        if kwargs['gpu']:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = 'cpu'

        l_gen_net = [self.latent_s + self.cond_in] + [kwargs['gen_w']] * kwargs['gen_l'] + [self.in_size]

        # Build the generator
        self.gen_net = []
        for l1, l2 in zip(l_gen_net[:-1], l_gen_net[1:]):
            self.gen_net += [nn.Linear(l1, l2), nn.ReLU()]
        self.gen_net.pop() # Regression problem, no activation function at the last layer
        self.gen = nn.Sequential(*self.gen_net)

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward(self, noise: torch.Tensor, context: torch.Tensor):
        """
        Define forward pass
        :param noise: noise torch tensor
        :param context: conditional torch tensor
        :return: output torch tensor
        """

        pred = self.gen(torch.cat((noise, context), dim=1))

        return pred

    def weights_initialize(self, mean: float, std: float):
        """
        Initialize self model parameters following a normal distribution based on mean and std
        :param mean: mean of the standard distribution
        :param std : standard deviation of the normal distribution
        :return: None
        """
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, mean=mean, std=std)

    def sample(self, n_s=1, x_cond:np.array=None):
        """
        :param n_s: number of scenarios
        :param x_cond: context (weather forecasts, etc) into an array of shape (self.cond_in,)
        :return: samples into an array of shape (nb_samples, self.in_size)
        """
        # Generate samples from a multivariate Gaussian
        z = torch.randn(n_s, self.latent_s).to(self.device)
        context = torch.tensor(np.tile(x_cond, n_s).reshape(n_s, self.cond_in)).to(self.device).float()
        scenarios = self.gen(torch.cat((z, context), dim=1)).view(n_s, -1).cpu().detach().numpy()

        return scenarios


class Discriminator_wassertein(nn.Module):
    """
    Define critic class, discriminator using Wasserstein distance estimate.
    Return a positive number. Higher the output is, more realistic is the input.
    """

    def __init__(self, **kwargs):

        """
        Critic constructor
        :param input_dim: size of the input (real or fake) torch tensor
        :param condition_dim: size ot the conditional torch tensor
        :param hidden_layer: number of hidden layer
        :param neurons_per_layer : number of neurons per hidden layer
        """
        super(Discriminator_wassertein, self).__init__()

        self.in_size = kwargs['in_size']   # Dim of the random variable to model (PV, wind power, etc)
        self.cond_in = kwargs['cond_in']   # Dim of context (weather forecasts, etc)
        self.latent_s = kwargs['latent_s'] # Dim of the latent space
        self.lambda_gp = kwargs['lambda_gp']

        # Set GPU if available
        if kwargs['gpu']:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = 'cpu'

        l_dis_net = [self.in_size + self.cond_in] + [kwargs['gen_w']] * kwargs['gen_l'] + [1]

        # Build the discriminator
        alpha = 0.01
        self.dis_net = []
        for l1, l2 in zip(l_dis_net[:-1], l_dis_net[1:]):
            self.dis_net += [nn.Linear(l1, l2), nn.LeakyReLU(alpha)]
        self.dis_net.pop() # The last activation function is a ReLU to return a positive number
        self.dis_net.append(nn.ReLU())
        self.dis = nn.Sequential(*self.dis_net)

    def loss(self, generated_samples: torch.Tensor, true_samples: torch.Tensor, context: torch.Tensor):

        # Discriminator's answers to generated and true samples
        D_true = self.dis(torch.cat((true_samples, context), dim=1))
        D_generated = self.dis(torch.cat((generated_samples, context), dim=1))
        # Compute Discriminator's loss with a gradient penalty to force Lipschitz condition
        gp = self.grad_pen(real=true_samples, samples=generated_samples, context=context)
        loss = -(torch.mean(D_true) - torch.mean(D_generated)) + self.lambda_gp * gp

        return loss


    def forward(self, input: torch.Tensor, context: torch.Tensor):
        """
        Define forward pass
        :param input: input (real or fake) torch tensor
        :param context: conditional torch tensor
        :return: output torch tensor
        """

        pred = self.dis(torch.cat((input, context), dim=1))

        return pred

    def weights_initialize(self, mean: float, std: float):
        """
        Initialize self model parameters following a normal distribution based on mean and std
        :param mean: mean of the standard distribution
        :param std : standard deviation of the normal distribution
        :return: None
        """
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, mean=mean, std=std)

    def grad_pen(self, real: torch.tensor, samples: torch.tensor, context: torch.Tensor):
        """
        For discriminator using Wasserstein distance estimate.
        Compute gradient penalty to add to critic loss in order to force Lipschitz condition using batch
        of interpolated sample. The Lipschitz constraint is obtained if the gradient is 1 on all interpolated
        sample. The gradient penalty forces this condition.
        :param real: batch of real samples, shape: batch_size * 24
        :param samples: batch of generated sample, shape: batch_size * 24
        :param context : batch of conditional torch tensor, shape: batch_size * 24
        :return: gradient penalty
        """

        # Interpolated sample
        bs, sample_size = real.shape[0], real.shape[1]
        epsilon = torch.rand((bs, sample_size), device=self.device)
        interpolated_sample = real * epsilon + samples * (1 - epsilon)
        # Compute critic scores
        mixed_score = self.dis(torch.cat((interpolated_sample, context), dim=1))
        # Gradient of the mixed_score with respect with the interpolated_sample
        gradient = torch.autograd.grad(inputs=interpolated_sample,
                                       outputs=mixed_score,
                                       grad_outputs=torch.ones_like(mixed_score),
                                       create_graph=True, retain_graph=True)[0]

        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_pen = torch.mean((gradient_norm - 1) ** 2)

        return gradient_pen


def fit_gan_wasserstein(nb_epoch: int, x_LS: np.array, y_LS: np.array, x_VS: np.array, y_VS: np.array, x_TEST: np.array, y_TEST: np.array, gen, dis, opt_gen, opt_dis, n_discriminator:int, batch_size:int=100, wdb:bool=False, gpu:bool=True):
    """
    Fit GAN with discriminator using the Wasserstein distance estimate.
    """
    # to assign the data to GPU with .to(device) on the data
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    # Assign models and data to gpu
    gen.to(device)
    dis.to(device)
    x_VS_gpu = torch.tensor(x_VS).to(device).float()
    y_VS_gpu = torch.tensor(y_VS).to(device).float()
    x_TEST_gpu = torch.tensor(x_TEST).to(device).float()
    y_TEST_gpu = torch.tensor(y_TEST).to(device).float()

    loss_list = []
    time_tot = 0.

    # WARNING: batch size = 10 % #LS
    batch_size = int(0.1 * y_LS.shape[0])

    for epoch in range(nb_epoch):
        start = timer()

        # Shuffle the data randomly at each epoch
        seed = random.randint(0, 2000)
        x_LS_shuffled, y_LS_shuffled = shuffle(x_LS, y_LS, random_state=seed)

        batch_dis_idx = 0
        batch_gen_idx = 0
        loss_D_batch = 0
        loss_G_batch = 0

        # Training batch loop
        batch_list = [i for i in range(batch_size, batch_size * y_LS.shape[0] // batch_size, batch_size)]
        for y_batch, x_batch in zip(np.split(y_LS_shuffled, batch_list), np.split(x_LS_shuffled, batch_list)):
            y_batch_LS = torch.tensor(y_batch).to(device).float()
            x_batch_LS = torch.tensor(x_batch).to(device).float()
            bs = x_batch_LS.shape[0]

            # 1. Train the Discriminator
            # Critic wants to maximize : E(C(x)) - E(C(G(z)))
            #             <~> maximize : mean(C(x)) - mean(C(G(z)))
            #             <-> minimize : -{mean(C(x)) - mean(C(G(z)))}

            # Generated samples
            G_LS_samples = gen(noise=torch.randn(bs, gen.latent_s).to(device), context=x_batch_LS)
            # Compute Discriminator's loss
            loss_D = dis.loss(generated_samples=G_LS_samples, true_samples=y_batch_LS, context=x_batch_LS)
            loss_D_batch += loss_D.detach()
            # Update critic's weight
            opt_dis.zero_grad()
            loss_D.backward()
            opt_dis.step()

            # N_CRITIC update for discriminator while one for generator
            # 2. Train the Generator
            if ((batch_dis_idx + 1) % n_discriminator) == 0:
                # Train Generator
                # Generator has the opposed objective that of critic :
                #      wants to minimize : E(C(x)) - E(C(G(z)))
                #           <-> minimize : - E(C(G(z)))
                #           <-> minimize : -(mean(C(G(z)))
                # Generated samples
                G_LS_samples = gen(noise=torch.randn(bs, gen.latent_s).to(device), context=x_batch_LS)
                D_LS = dis(input=G_LS_samples, context=x_batch_LS)
                # Compute generator's loss
                lossG = -torch.mean(D_LS)
                loss_G_batch += lossG.detach()
                # Update generator's weight
                opt_gen.zero_grad()
                lossG.backward()
                opt_gen.step()
                batch_gen_idx += 1

            batch_dis_idx += 1

        # LS loss is the average over all the batch
        loss_D_LS = loss_D_batch / batch_dis_idx
        loss_G_LS = loss_G_batch / batch_gen_idx

        # VS loss
        # D
        G_VS_samples = gen(noise=torch.randn(y_VS_gpu.shape[0], gen.latent_s).to(device), context=x_VS_gpu)
        loss_D_VS = dis.loss(generated_samples=G_VS_samples, true_samples=y_VS_gpu, context=x_VS_gpu).detach()
        # G
        D_VS = dis(input=G_VS_samples, context=x_VS_gpu)
        loss_G_VS = -torch.mean(D_VS).detach()

        # TEST loss
        # D
        G_TEST_samples = gen(noise=torch.randn(y_TEST.shape[0], gen.latent_s).to(device), context=x_TEST_gpu)
        loss_D_TEST = dis.loss(generated_samples=G_TEST_samples, true_samples=y_TEST_gpu, context=x_TEST_gpu).detach()

        # G
        D_TEST = dis(input=G_TEST_samples, context=x_TEST_gpu)
        loss_G_TEST = -torch.mean(D_TEST).detach()

        # Save NF model when the VS loss is minimal
        loss_list.append([loss_D_LS, loss_G_LS, loss_D_VS, loss_G_VS, loss_D_TEST, loss_G_TEST])

        end = timer()
        time_tot += end - start

        if wdb:
            wandb.log({"D ls loss": loss_D_LS})
            wandb.log({"G ls loss": loss_G_LS})
            wandb.log({"D vs loss": loss_D_VS})
            wandb.log({"G vs loss": loss_G_VS})
            wandb.log({"D test loss": loss_D_TEST})
            wandb.log({"G test loss": loss_G_TEST})

        if epoch % 10 == 0:
            print("Epoch {:.0f} Approximate time left : {:2f} min - D LS loss: {:4f} G LS loss: {:4f} D VS loss: {:4f} G VS loss: {:4f}".format(epoch, time_tot / (epoch + 1) * (nb_epoch - (epoch + 1)) / 60, loss_D_LS, loss_G_LS, loss_D_VS, loss_G_VS), end="\r", flush=True)
    print('Fitting time_tot %.0f min' %(time_tot/60))

    return np.asarray(torch.tensor(loss_list, device='cpu')), gen, dis

def build_gan_scenarios(n_s: int, x: np.array, y_scaler, gen, max:int=1, gpu:bool=True, tag:str= 'pv', non_null_indexes:list=[]):
    """
    Build scenarios for a VAE multi-output (VS or TEST sets).
    Scenarios are generated into an array (n_periods, n_s) where n_periods = 24 * n_days
    :return: scenarios (n_periods, n_s)
    """

    # to assign the data to GPU with .to(device) on the data
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    gen.to(device)

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
        predictions = gen.sample(n_s=n_s, x_cond=x[i, :])
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


def plot_GAN_loss(loss: np.array, nb_days: list, ylim: list, dir_path: str, name: str):
    """
    Plot the loss vs epoch.
    """
    FONTSIZE = 10
    nb_epoch = loss.shape[0]
    epoch_min_D = np.nanargmin(loss[:, 2])
    epoch_min_G = np.nanargmin(loss[:, 3])

    plt.figure()
    plt.plot(loss[:, 0], label='D LS')
    plt.plot(loss[:, 1], label='G LS')
    plt.plot(loss[:, 2], label='D VS')
    plt.plot(loss[:, 3], label='G VS')
    plt.plot(loss[:, 4], label='D TEST')
    plt.plot(loss[:, 5], label='G TEST')
    plt.hlines(y=0, xmin=0, xmax=nb_epoch)
    # plt.vlines(x=epoch_min_D, ymin=ylim[0], ymax=ylim[1], colors='k', label='D VS loss at ' + str(epoch_min_D) + ' = ' + str(round(loss[epoch_min_D, 2], 2)))
    # plt.vlines(x=epoch_min_G, ymin=ylim[0], ymax=ylim[1], colors='k', label='G VS loss at ' + str(epoch_min_G) + ' = ' + str(round(loss[epoch_min_G, 3], 2)))
    plt.xlabel('epoch', fontsize=FONTSIZE)
    plt.ylabel('ll loss', fontsize=FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlim(0, nb_epoch)
    plt.ylim(ylim[0], ylim[1])
    plt.title(name+'#LS ' + str(nb_days[0])+' #LS ' + str(nb_days[0])+' #TEST ' + str(nb_days[2]))
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dir_path + name + '.pdf')
    plt.show()

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())

