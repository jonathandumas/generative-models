"""
Implements Masked Autoregressive network
Andrej Karpathy's implementation of based on https://arxiv.org/abs/1502.03509
Modified by Antoine Wehenkel
"""

import numpy as np
from .Conditioner import Conditioner
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MAN(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=False, random=False, device="cpu"):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """

        super().__init__()
        self.random = random
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        #assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
        if self.m and self.num_masks == 1: return  # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        if self.random:
            self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
            for l in range(L):
                self.m[l] = rng.randint(self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l])
        else:
            self.m[-1] = np.arange(self.nin)
            for l in range(L):
                self.m[l] = np.array([self.nin - 1 - (i % self.nin) for i in range(self.hidden_sizes[l])])

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

        # map between in_d and order
        self.i_map = self.m[-1].copy()
        for k in range(len(self.m[-1])):
            self.i_map[self.m[-1][k]] = k

    def forward(self, x):
        return self.net(x).view(x.shape[0], -1, x.shape[1]).permute(0, 2, 1)


# ------------------------------------------------------------------------------


class ConditionnalMAN(MAN):

    def __init__(self, nin, cond_in, hidden_sizes, nout, num_masks=1, natural_ordering=False, random=False,
                 device="cpu"):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """

        super().__init__(nin + cond_in, hidden_sizes, nout, num_masks, natural_ordering, random, device)
        self.nin_non_cond = nin
        self.cond_in = cond_in

    def forward(self, x, context):
        if context is not None:
            out = super().forward(torch.cat((context, x), 1))
        else:
            out = super().forward(x)
        out = out.contiguous()[:, self.cond_in:, :]
        return out


class AutoregressiveConditioner(Conditioner):
    """
    in_size: The dimension of the input vector, this corresponds to the number of autoregressive output vectors.
    hidden: The dimension of the masked autoregressive neural network hidden layers.
    out_size: The dimension of the output vectors.
    cond_in: The dimension of the additional context input.
    """
    def __init__(self, in_size, hidden, out_size, cond_in=0):
        super(AutoregressiveConditioner, self).__init__()
        self.in_size = in_size
        self.masked_autoregressive_net = ConditionnalMAN(in_size, cond_in=cond_in, hidden_sizes=hidden, nout=out_size*(in_size + cond_in))
        self.register_buffer("A", 1 - torch.tril(torch.ones(in_size, in_size)).T)

    """
    x: An input tensor with dim=[b_size, in_size]
    context: A context/conditionning tensor with dim=[b_size, cond_in]
    return: An autoregressive embedding tensor of x conditionned on context, its dim is=[b_size, in_size, out_size]
    """
    def forward(self, x, context=None):
        return self.masked_autoregressive_net(x, context)

    def depth(self):
        return self.in_size - 1
