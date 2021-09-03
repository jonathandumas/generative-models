import torch.distributions as D
import torch
from math import pi
import torch.nn as nn


class FlowDensity(nn.Module):
    def __init__(self):
        super(FlowDensity, self).__init__()

    def forward(self, z):
        pass

    def sample(self, shape):
        pass


class NormalLogDensity(nn.Module):
    def __init__(self):
        super(NormalLogDensity, self).__init__()
        self.register_buffer("pi", torch.tensor(pi))

    def forward(self, z):
        return torch.distributions.Normal(loc=0., scale=1.).log_prob(z).sum(1)

    def sample(self, shape):
        return torch.randn(shape)


class MixtureLogDensity(nn.Module):
    def __init__(self, n_mode=10):
        super(MixtureLogDensity, self).__init__()
        self.register_buffer("pi", torch.tensor(pi))
        self.register_buffer("mu", torch.arange(-3., 3.0001, 6. / float(n_mode - 1)))
        self.register_buffer("sigma", torch.ones(n_mode, ) * 1.5 / float(n_mode))
        self.register_buffer("mix_weights", torch.ones(n_mode, ))

    def forward(self, z):
        mix = D.Categorical(self.mix_weights)
        comp = D.Normal(self.mu, self.sigma)
        dist = D.MixtureSameFamily(mix, comp)
        return dist.log_prob(z).sum(1)

