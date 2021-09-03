import torch
import torch.nn as nn
from models.Conditionners import Conditioner, DAGConditioner
from models.Normalizers import Normalizer
from models.Utils.Distributions import FlowDensity


class NormalizingFlow(nn.Module):
    def __init__(self):
        super(NormalizingFlow, self).__init__()

    '''
    Should return the x transformed and the log determinant of the Jacobian of the transformation
    '''
    def forward(self, x, context=None):
        pass

    '''
    Should return a term relative to the loss.
    '''
    def constraintsLoss(self):
        pass

    '''
    Should return the dagness of the associated graph.
    '''
    def DAGness(self):
        pass

    '''
    Step in the optimization procedure;
    '''
    def step(self, epoch_number, loss_avg):
        pass

    '''
    Return a list containing the conditioners.
    '''
    def getConditioners(self):
        pass

    '''
    Return True if the architecture is invertible.
    '''
    def isInvertible(self):
        pass

    '''
        Return a list containing the normalizers.
        '''

    def getNormalizers(self):
        pass

    '''
    Return the x that would generate z: [B, d] tensor.
    '''
    def invert(self, z, context=None):
        pass


class NormalizingFlowStep(NormalizingFlow):
    def __init__(self, conditioner: Conditioner, normalizer: Normalizer):
        super(NormalizingFlowStep, self).__init__()
        self.conditioner = conditioner
        self.normalizer = normalizer

    def forward(self, x, context=None):
        h = self.conditioner(x, context)
        z, jac = self.normalizer(x, h, context)
        return z, torch.log(jac).sum(1)

    def constraintsLoss(self):
        if issubclass(type(self.conditioner), DAGConditioner):
            return self.conditioner.loss()
        return 0.

    def DAGness(self):
        if issubclass(type(self.conditioner), DAGConditioner):
            return [self.conditioner.get_power_trace()]
        return [0.]

    def step(self, epoch_number, loss_avg):
        if issubclass(type(self.conditioner), DAGConditioner):
            self.conditioner.step(epoch_number, loss_avg)

    def getConditioners(self):
        return [self.conditioner]

    def getNormalizers(self):
        return [self.normalizer]

    def isInvertible(self):
        for conditioner in self.getConditioners():
            if not conditioner.is_invertible:
                return False
        return True

    def invert(self, z, context=None):
        x = torch.zeros_like(z)
        for i in range(self.conditioner.depth() + 1):
            h = self.conditioner(x, context)
            x_prev = x
            x = self.normalizer.inverse_transform(z, h, context)
            if torch.norm(x - x_prev) == 0.:
                break
        return x


class FCNormalizingFlow(NormalizingFlow):
    def __init__(self, steps: NormalizingFlow, z_log_density: FlowDensity):
        super(FCNormalizingFlow, self).__init__()
        self.steps = nn.ModuleList()
        self.z_log_density = z_log_density
        for step in steps:
            self.steps.append(step)

    def forward(self, x, context=None):
        jac_tot = 0.
        inv_idx = torch.arange(x.shape[1] - 1, -1, -1).long()
        for step in self.steps:
            z, jac = step(x, context)
            x = z[:, inv_idx]
            jac_tot += jac

        return z, jac_tot

    def constraintsLoss(self):
        loss = 0.
        for step in self.steps:
                loss += step.constraintsLoss()
        return loss

    def DAGness(self):
        dagness = []
        for step in self.steps:
            dagness += step.DAGness()
        return dagness

    def step(self, epoch_number, loss_avg):
        for step in self.steps:
            step.step(epoch_number, loss_avg)

    def loss(self, z, jac):
        log_p_x = jac + self.z_log_density(z)
        return self.constraintsLoss() - log_p_x.mean()

    def compute_ll(self, x, context=None):
        z, jac_tot = self(x, context)
        log_p_x = jac_tot + self.z_log_density(z)
        return log_p_x, z

    def getNormalizers(self):
        normalizers = []
        for step in self.steps:
            normalizers += step.getNormalizers()
        return normalizers

    def getConditioners(self):
        conditioners = []
        for step in self.steps:
            conditioners += step.getConditioners()
        return conditioners

    def isInvertible(self):
        for conditioner in self.getConditioners():
            if not conditioner.is_invertible:
                return False
        return True

    def invert(self, z, context=None):
        if type(z) is list:
            z = self.z_log_density.sample(z)
        inv_idx = torch.arange(z.shape[1] - 1, -1, -1).long()
        for step in range(len(self.steps)):
            x = self.steps[-step - 1].invert(z, context)
            z = x[:, inv_idx]
        return x


class CNNormalizingFlow(FCNormalizingFlow):
    def __init__(self, steps, z_log_density, dropping_factors):
        super(CNNormalizingFlow, self).__init__(steps, z_log_density)
        self.dropping_factors = dropping_factors

    def forward(self, x, context=None):
        b_size = x.shape[0]
        jac_tot = 0.
        z_all = []
        for step, drop_factors in zip(self.steps, self.dropping_factors):
            z, jac = step(x, context)
            d_c, d_h, d_w = drop_factors
            C, H, W = step.img_sizes
            c, h, w = int(C/d_c), int(H/d_h), int(W/d_w)
            z_reshaped = z.view(-1, C, H, W).unfold(1, d_c, d_c).unfold(2, d_h, d_h) \
                    .unfold(3, d_w, d_w).contiguous().view(b_size, c, h, w, -1)
            z_all += [z_reshaped[:, :, :, :, 1:].contiguous().view(b_size, -1)]
            x = z.view(-1, C, H, W).unfold(1, d_c, d_c).unfold(2, d_h, d_h) \
                    .unfold(3, d_w, d_w).contiguous().view(b_size, c, h, w, -1)[:, :, :, :, 0] \
                .contiguous().view(b_size, -1)
            jac_tot += jac
        z_all += [x]
        z = torch.cat(z_all, 1)
        return z, jac_tot

    def invert(self, z, context=None):
        b_size = z.shape[0]
        z_all = []
        i = 0
        for step, drop_factors in zip(self.steps, self.dropping_factors):
            d_c, d_h, d_w = drop_factors
            C, H, W = step.img_sizes
            c, h, w = int(C / d_c), int(H / d_h), int(W / d_w)
            nb_z = C*H*W - c*h*w if C*H*W != c*h*w else c*h*w
            z_all += [z[:, i:i+nb_z]]
            i += nb_z

        x = 0.
        for i in range(1, len(self.steps) + 1):
            step = self.steps[-i]
            drop_factors = self.dropping_factors[-i]
            d_c, d_h, d_w = drop_factors
            C, H, W = step.img_sizes
            c, h, w = int(C / d_c), int(H / d_h), int(W / d_w)
            z = z_all[-i]
            if c*h*w != C*H*W:
                z = z.view(b_size, c, h, w, -1)
                x = x.view(b_size, c, h, w, 1)
                z = torch.cat((x, z), 4)
                z = z.view(b_size, c, h, w, d_c, d_h, d_w)
                z = z.permute(0, 1, 2, 3, 6, 4, 5).contiguous().view(b_size, c, h, W, d_c, d_h)
                z = z.permute(0, 1, 2, 5, 3, 4).contiguous().view(b_size, c, H, W, d_c)
                z = z.permute(0, 1, 4, 2, 3).contiguous().view(b_size, C, H, W)
            x = step.invert(z.view(b_size, -1), context)
        return x


class FixedScalingStep(NormalizingFlow):
    def __init__(self, mu, std):
        super(FixedScalingStep, self).__init__()
        self.mu = mu
        self.std = std

    def forward(self, x, context=None):
        z = (x - self.mu.unsqueeze(0).expand(x.shape[0], -1))/self.std.unsqueeze(0).expand(x.shape[0], -1)
        jac = self.std.unsqueeze(0).expand(x.shape[0], -1)
        return z, -torch.log(jac).sum(1)

    def constraintsLoss(self):
        return 0.

    def DAGness(self):
        return [0.]

    def step(self, epoch_number, loss_avg):
        return

    def getConditioners(self):
        return []

    def getNormalizers(self):
        return []

    def isInvertible(self):
        return True

    def invert(self, z, context=None):
        x = z * self.std.unsqueeze(0).expand(z.shape[0], -1) + self.mu.unsqueeze(0).expand(z.shape[0], -1)
        return x
