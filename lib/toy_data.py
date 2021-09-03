# <Source: https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py >

import numpy as np
import sklearn
import torch
import sklearn.datasets
from PIL import Image
import os

# Dataset iterator
def inf_train_gen(data, rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()
        #rng = np.random.RandomState(42)
        #print(rng)

    if data == "1Normal":
        z_dist = torch.distributions.Normal(loc=2, scale=.5)
        z = z_dist.sample((batch_size, 1))
        return z

    if data == "2spirals-8gaussians":
        data1 = inf_train_gen("2spirals", rng=rng, batch_size=batch_size)
        data2 = inf_train_gen("8gaussians", rng=rng, batch_size=batch_size)
        return np.concatenate([data1, data2], axis=1)

    if data == "4-2spirals-8gaussians":
        data1 = inf_train_gen("2spirals", rng=rng, batch_size=batch_size)
        data2 = inf_train_gen("8gaussians", rng=rng, batch_size=batch_size)
        data3 = inf_train_gen("2spirals", rng=rng, batch_size=batch_size)
        data4 = inf_train_gen("8gaussians", rng=rng, batch_size=batch_size)
        return np.concatenate([data1, data2, data3, data4], axis=1)

    if data == "8-2spirals-8gaussians":
        data1 = inf_train_gen("4-2spirals-8gaussians", rng=rng, batch_size=batch_size)
        data2 = inf_train_gen("4-2spirals-8gaussians", rng=rng, batch_size=batch_size)
        return np.concatenate([data1, data2], axis=1)

    if data == "3-MIX_DEP":
        data1 = inf_train_gen("2spirals", rng=rng, batch_size=batch_size)
        data2 = inf_train_gen("8gaussians", rng=rng, batch_size=batch_size)
        data3 = inf_train_gen("circles", rng=rng, batch_size=batch_size)
        p = rng.randint(0, 2, batch_size).reshape(-1, 1)
        p = np.concatenate([p, p], axis=1)
        dist = 2*((data1[:, 0]**2 + data1[:, 1]**2)**.5).reshape(-1, 1) - .5
        data3 = np.concatenate([dist, dist], axis=1) * data2*.5 + rng.randn(batch_size, 2) * .1
        return np.concatenate([data1, data2, data3], axis=1).astype(data2.dtype)

    if data == "7-pyramid":
        data1 = inf_train_gen("circles", rng=rng, batch_size=batch_size)
        data2 = inf_train_gen("8gaussians", rng=rng, batch_size=batch_size)
        data3 = rng.randn(batch_size, 1) + data1.max(1).reshape(-1, 1)
        data4 = rng.randn(batch_size, 1) + data2.min(1).reshape(-1, 1)
        p = rng.randint(0, 2, batch_size).reshape(-1, 1)
        data5 = p * np.sin(data3 + data4) + (1 - p) * np.cos(data3 + data4) + rng.randn(batch_size, 1)
        return np.concatenate([data1, data2, data3, data4, data5], axis=1).astype(data2.dtype)

    if data == "8-MIX":
        data1 = inf_train_gen("2spirals", rng=rng, batch_size=batch_size)
        data2 = inf_train_gen("8gaussians", rng=rng, batch_size=batch_size)
        data3 = inf_train_gen("swissroll", rng=rng, batch_size=batch_size)
        data4 = inf_train_gen("circles", rng=rng, batch_size=batch_size)
        data8 = inf_train_gen("moons", rng=rng, batch_size=batch_size)
        data6 = inf_train_gen("pinwheel", rng=rng, batch_size=batch_size)
        data7 = inf_train_gen("checkerboard", rng=rng, batch_size=batch_size)
        data5 = inf_train_gen("line", rng=rng, batch_size=batch_size)
        std = np.array([1.604934 , 1.584863 , 2.0310535, 2.0305095, 1.337718 , 1.4043778,  1.6944685, 1.6935346,
                        1.7434783, 1.0092416, 1.4860426, 1.485661 , 2.3067558, 2.311637 , 1.4430547, 1.4430547], dtype=np.float32)
        data = np.concatenate([data1, data2, data3, data4, data5, data6, data7, data8], axis=1)

        return data/std
    if data == "7-MIX":
        data1 = inf_train_gen("2spirals", rng=rng, batch_size=batch_size)
        data2 = inf_train_gen("8gaussians", rng=rng, batch_size=batch_size)
        data3 = inf_train_gen("swissroll", rng=rng, batch_size=batch_size)
        data4 = inf_train_gen("circles", rng=rng, batch_size=batch_size)
        data5 = inf_train_gen("moons", rng=rng, batch_size=batch_size)
        data6 = inf_train_gen("pinwheel", rng=rng, batch_size=batch_size)
        data7 = inf_train_gen("checkerboard", rng=rng, batch_size=batch_size)
        std = np.array([1.604934 , 1.584863 , 2.0310535, 2.0305095, 1.337718 , 1.4043778,  1.6944685, 1.6935346,
                        1.7434783, 1.0092416, 1.4860426, 1.485661 , 2.3067558, 2.311637], dtype=np.float32)
        data = np.concatenate([data1, data2, data3, data4, data5, data6, data7], axis=1)

        return data/std


    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        data = data.astype("float32")
        return data

    elif data == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "2gaussians":
        scale = 4.
        centers = [(.5, -.5), (-.5, .5)]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * .75
            idx = rng.randint(2)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        #dataset /= 1.414
        return dataset

    elif data == "4gaussians":
        scale = 4.
        centers = [(.5, -.5), (-.5, .5), (.5, .5), (-.5, -.5)]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * .75
            idx = rng.randint(4)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        # dataset /= 1.414
        return dataset

    elif data == "2igaussians":
        scale = 4.
        centers = [(.5, 0.), (-.5, .0)]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * .75
            idx = rng.randint(2)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        # dataset /= 1.414
        return dataset

    elif data == "conditionnal8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        context = np.zeros((batch_size, 8))
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            context[i, idx] = 1
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset, context

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations)).astype("float32")

    elif data == "2spirals":
        n = np.sqrt(rng.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + rng.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + rng.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += rng.randn(*x.shape) * 0.1
        return x.astype("float32")

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1).astype("float32") * 2

    elif data == "line":
        x = rng.rand(batch_size)
        #x = np.arange(0., 1., 1/batch_size)
        x = x * 5 - 2.5
        y = x #- x + rng.rand(batch_size)
        return np.stack((x, y), 1).astype("float32")
    elif data == "line-noisy":
        x = rng.rand(batch_size)
        x = x * 5 - 2.5
        y = x + rng.randn(batch_size)
        return np.stack((x, y), 1).astype("float32")
    elif data == "cos":
        x = rng.rand(batch_size) * 6 - 3
        y = np.sin(x*5) * 2.5 + np.random.randn(batch_size) * .3
        return np.stack((x, y), 1).astype("float32")
    elif data == "joint_gaussian":
        x2 = torch.distributions.Normal(0., 4.).sample((batch_size, 1))
        x1 = torch.distributions.Normal(0., 1.).sample((batch_size, 1)) + (x2**2)/4

        return torch.cat((x1, x2), 1)
    elif data == "woodStructural":
        log_lik = 0.
        z0 = torch.distributions.log_normal.LogNormal(5., 1., validate_args=None).sample((batch_size, 1))
        log_lik += torch.distributions.log_normal.LogNormal(5., 1., validate_args=None).log_prob(z0)

        z1 = torch.distributions.log_normal.LogNormal(-2., 1., validate_args=None).sample((batch_size, 1))
        log_lik += torch.distributions.log_normal.LogNormal(-2., 1., validate_args=None).log_prob(z1)

        z2 = torch.distributions.Normal(loc=torch.tanh(z0 + z1 - 2.8), scale=.1).sample()
        log_lik += torch.distributions.Normal(loc=torch.tanh(z0 + z1 - 2.8), scale=.1).log_prob(z2)

        z3 = torch.distributions.Normal(loc=z0 * z1, scale=.1).sample()
        log_lik += torch.distributions.Normal(loc=z0 * z1, scale=.1).log_prob(z3)

        z4 = torch.distributions.Normal(loc=7, scale=2.).sample((batch_size, 1))
        log_lik += torch.distributions.Normal(loc=7, scale=2.).log_prob(z4)

        z5 = torch.distributions.Normal(loc=torch.tanh(z3 + z4), scale=.1).sample()
        log_lik += torch.distributions.Normal(loc=torch.tanh(z3 + z4), scale=.1).log_prob(z5)

        x0 = torch.distributions.Normal(loc=z3, scale=.1).sample()
        log_lik += torch.distributions.Normal(loc=z3, scale=.1).log_prob(x0)

        x1 = torch.distributions.Normal(loc=z5, scale=.1).sample()
        log_lik += torch.distributions.Normal(loc=z5, scale=.1).log_prob(x1)

        z = torch.cat((z0, z1, z2, z3, z4, z5, x0, x1), 1)
        #print(log_lik.mean() + torch.log(z.std(0)).sum())
        #exit()
        return z
    else:
        return inf_train_gen("8gaussians", rng, batch_size)


def getA(toy):
    if toy == "8-MIX":
        A = torch.zeros(16, 16)
        for i in range(8):
            A[i * 2, 2*i+1] = 1
    elif toy == "7-MIX":
        A = torch.zeros(14, 14)
        for i in range(7):
            A[i * 2, 2 * i + 1] = 1

    elif toy == "3-MIX_DEP":
        A = torch.zeros(6, 6)
        for i in range(2):
            A[i * 2, 2 * i + 1] = 1
        A[4, 0] = 1
        A[4, 2] = 1
        A[4, 1] = 1
        A[4, 3] = 1
        A[5, 0] = 1
        A[5, 2] = 1
        A[5, 1] = 1
        A[5, 3] = 1
    elif toy == "woodStructural":
        A = torch.zeros(8, 8)
        A[2, 0] = 1
        A[2, 1] = 1
        A[3, 0] = 1
        A[3, 1] = 1
        A[5, 3] = 1
        A[5, 4] = 1
        A[6, 3] = 1
        A[7, 5] = 1
    elif toy == "7-pyramid":
        A = torch.zeros(7, 7)
        A[0, 1] = 1
        A[2, 3] = 1
        A[4, 0] = 1
        A[4, 1] = 1
        A[5, 2] = 1
        A[5, 3] = 1
        A[6, 4] = 1
        A[6, 5] = 1
    elif toy == '1Normal':
        A = torch.zeros(1, 1)
    else:
        return None
    return A



