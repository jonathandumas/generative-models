import torch
import torch.nn as nn
from .DAGConditioner import DAGConditioner


def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


class DAGMLP(nn.Module):
    def __init__(self, in_size, hidden, out_size, cond_in=0):
        super(DAGMLP, self).__init__()
        in_size = in_size
        l1 = [in_size + cond_in] + hidden
        l2 = hidden + [out_size]
        layers = []
        for h1, h2 in zip(l1, l2):
            layers += [nn.Linear(h1, h2), nn.ReLU()]
        layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SubDAGConditioner(DAGConditioner):
    def __init__(self, in_size, hidden, out_size, cond_in=0, soft_thresholding=True, h_thresh=0., gumble_T=1.,
                 hot_encoding=False, l1=0., nb_epoch_update=1, A_prior=None, sub_mask=None):
        super(SubDAGConditioner, self).__init__(in_size, hidden, out_size, cond_in, soft_thresholding, h_thresh,
                                                gumble_T, hot_encoding, l1, nb_epoch_update, A_prior)

        self.register_buffer("sub_mask", sub_mask)

    def forward(self, x, context=None):
        b_size = x.shape[0]
        if self.h_thresh > 0:
            if self.stoch_gate:
                e = (x.unsqueeze(1).expand(-1, self.in_size, -1) * self.stochastic_gate(self.hard_thresholded_A().unsqueeze(0)
                                                                                        .expand(x.shape[0], -1, -1)))
            elif self.noise_gate:
                e = self.noiser_gate(x.unsqueeze(1).expand(-1, self.in_size, -1),
                                     self.hard_thresholded_A().unsqueeze(0)
                                     .expand(x.shape[0], -1, -1))
            else:
                e = (x.unsqueeze(1).expand(-1, self.in_size, -1) * self.hard_thresholded_A().unsqueeze(0)
                     .expand(x.shape[0], -1, -1))
        elif self.s_thresh:
            if self.stoch_gate:
                e = (x.unsqueeze(1).expand(-1, self.in_size, -1) * self.stochastic_gate(self.soft_thresholded_A().unsqueeze(0)
                                                                                        .expand(x.shape[0], -1, -1)))
            elif self.noise_gate:
                e = self.noiser_gate(x.unsqueeze(1).expand(-1, self.in_size, -1),
                                     self.soft_thresholded_A().unsqueeze(0).expand(x.shape[0], -1, -1))
            else:
                e = (x.unsqueeze(1).expand(-1, self.in_size, -1) * self.soft_thresholded_A().unsqueeze(0)
                     .expand(x.shape[0], -1, -1))
        else:
            e = (x.unsqueeze(1).expand(-1, self.in_size, -1) * self.A.unsqueeze(0).expand(x.shape[0], -1, -1))

        hot_encoding = torch.eye(self.in_size, device=self.A.device).unsqueeze(0).expand(x.shape[0], -1, -1)\
            .contiguous().view(-1, self.in_size)
        # TODO CLEAN CODE FOR the positional encoding.
        width = int(self.in_size**.5)
        indices = torch.arange(width, device=self.A.device).unsqueeze(0).expand(width, -1).contiguous()
        mesh = torch.cat((indices.view(-1, 1), indices.T.contiguous().view(-1, 1)), 1).float()/width
        pos_encoding = mesh.unsqueeze(0).expand(x.shape[0], -1, -1).contiguous().view(-1, 2)
        mask_size = self.sub_mask.shape[1]
        e = batched_index_select(e.view(x.shape[0] * self.in_size, -1), 1,
                                 self.sub_mask.unsqueeze(0).expand(b_size, -1, -1).contiguous().view(-1, mask_size))
        if context is not None:
            context = context.unsqueeze(1).expand(-1, self.in_size, -1).reshape(b_size*self.in_size, -1)
            context = torch.cat((pos_encoding, context), 1)
            e = self.embedding_net(e, context=context)
        else:
            e = self.embedding_net(e, context=pos_encoding)
        #full_e = torch.cat((e, hot_encoding), 1).view(x.shape[0], self.in_size, -1)
        full_e = torch.cat((e, pos_encoding), 1).reshape(x.shape[0], self.in_size, -1)
        # TODO Add context
        return full_e
