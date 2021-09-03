# <Source: https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py >

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

LOW = -2
HIGH = 2

font = {'size': 22}

matplotlib.rc('font', **font)


def plt_potential_func(potential, ax, npts=100, title="$p(x)$"):
    """
    Args:
        potential: computes U(z_k) given z_k
    """
    xside = np.linspace(LOW, HIGH, npts)
    yside = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(xside, yside)
    z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    z = torch.Tensor(z)
    u = potential(z).cpu().numpy()
    p = np.exp(-u).reshape(npts, npts)

    plt.pcolormesh(xx, yy, p)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow_2D(transform, ax, npts=50, range_xy=[[-1, 1], [-1, 1]], title="$q(x)$", device="cpu"):
    """
    Args:
        transform: computes z_k and log(q_k) given z_0
    """
    [low_x, high_x], [low_y, high_y] = range_xy
    side_x = np.linspace(low_x, high_x, npts)
    side_y = np.linspace(low_y, high_y, npts)
    xx, yy = np.meshgrid(side_x, side_y)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    with torch.no_grad():
        logqx, z = transform(torch.tensor(x).float().to(device))

    qz = logqx.exp().cpu().numpy().reshape(npts, npts)
    qz_1 = qz.sum(1)
    qz_2 = qz.sum(0)

    pcol = ax.pcolormesh(xx, yy, qz, shading='auto')
    pcol.set_edgecolor('face')
    ax.set_xlim(low_x, high_x)
    ax.set_ylim(low_y, high_y)

    return qz_1, qz_2

def plt_stream(transform, ax, npts=200, title="Density streamflow", device="cpu"):
    side = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(side, side)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    with torch.no_grad():
        logqx, z = transform(torch.tensor(x).float().to(device))
    d_z_x = -(x - z.cpu().numpy())[:, 0].reshape(xx.shape)
    d_z_y = -(x - z.cpu().numpy())[:, 1].reshape(xx.shape)
    plt.streamplot(xx, yy, d_z_x, d_z_y, color=(d_z_y**2 + d_z_x**2)/2, cmap='autumn')


def plt_flow_density(prior_logdensity, inverse_transform, ax, npts=100, memory=100, title="$q(x)$", device="cpu"):
    side = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(side, side)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    x = torch.from_numpy(x).type(torch.float32).to(device)
    zeros = torch.zeros(x.shape[0], 1).to(x)

    z, delta_logp = [], []
    inds = torch.arange(0, x.shape[0]).to(torch.int64)
    for ii in torch.split(inds, int(memory**2)):
        z_, delta_logp_ = inverse_transform(x[ii], zeros[ii])
        z.append(z_)
        delta_logp.append(delta_logp_)
    z = torch.cat(z, 0)
    delta_logp = torch.cat(delta_logp, 0)

    logpz = prior_logdensity(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    px = np.exp(logpx.cpu().numpy()).reshape(npts, npts)

    ax.imshow(px)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow_samples(prior_sample, transform, ax, npts=200, memory=100, title="$x \sim q(x)$", device="cpu"):
    z = prior_sample(npts * npts, 2).type(torch.float32).to(device)
    zk = []
    inds = torch.arange(0, z.shape[0]).to(torch.int64)
    for ii in torch.split(inds, int(memory**2)):
        zk.append(transform(z[ii]))
    zk = torch.cat(zk, 0).cpu().numpy()
    ax.hist2d(zk[:, 0], zk[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=npts)
    #ax.invert_yaxis()
    ax.get_xaxis().set_ticks(np.arange(LOW, HIGH + .1, (HIGH - LOW)/4))
    ax.get_yaxis().set_ticks(np.arange(LOW, HIGH + .1, (HIGH - LOW)/4))
    ax.set_title(title)


def plt_samples(samples, ax, npts=200, title="$x \sim p(x)$"):
    #print(samples.min(0), samples.max(0), samples.mean(0))
    ax.hist2d(samples[:, 0], samples[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=npts, density=True)
    ax.get_xaxis().set_ticks(np.arange(LOW, HIGH + .1, (HIGH - LOW)/4))
    ax.get_yaxis().set_ticks(np.arange(LOW, HIGH + .1, (HIGH - LOW)/4))
    ax.set_title(title)


def visualize_transform(
    potential_or_samples, prior_sample, prior_density, transform=None, inverse_transform=None, samples=True, npts=100,
    memory=100, device="cpu"
):
    """Produces visualization for the model density and samples from the model."""
    plt.clf()
    ax = plt.subplot(1, 3, 1, aspect="equal")
    if samples:
        plt_samples(potential_or_samples, ax, npts=npts)
    else:
        plt_potential_func(potential_or_samples, ax, npts=npts)

    ax = plt.subplot(1, 3, 2, aspect="equal")
    if inverse_transform is None:
        plt_flow(prior_density, transform, ax, npts=npts, device=device)
    else:
        plt_flow_density(prior_density, inverse_transform, ax, npts=npts, memory=memory, device=device)

    ax = plt.subplot(1, 3, 3, aspect="equal")
    if transform is not None:
        plt_flow_samples(prior_sample, transform, ax, npts=npts, memory=memory, device=device)
