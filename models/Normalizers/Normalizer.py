import torch.nn as nn


class Normalizer(nn.Module):
    def __init__(self):
        super(Normalizer, self).__init__()

    '''
    forward(self, x, context=None):
    :param x: A tensor [B, d]
    :param h: A tensor [B, d, h]
    :param context: A tensor [B, c]
    :return: z: [B, d] x transformed by a one-to-one mapping conditioned on h.
             jac: [B, d] the diagonal terms of the Jacobian.
    '''
    def forward(self, x, h, context=None):
        pass


    '''
    inverse_transform(self, z, h, context=None):
    :param z: A tensor [B, d]
    :param h: A tensor [B, d, h]
    :param context: A tensor [B, c]
    :return x: [B, d] the x that would generate z given the embedding and context.
    '''
    def inverse_transform(self, z, h, context=None):
        pass
