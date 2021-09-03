import torch.nn as nn
import torch
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_d, hidden, out_d, act_f=nn.ReLU()):
        super().__init__()
        self.in_d = in_d
        self.hiddens = hidden
        self.out_d = out_d
        self.act_f = act_f
        layers_dim = [in_d] + hidden + [out_d]
        layers = []
        for dim_in, dim_out in zip(layers_dim[:-1], layers_dim[1:]):
            layers += [nn.Linear(dim_in, dim_out), act_f]
        layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, x, context=None):
        if context is not None:
            x = torch.cat((x, context), 1)
        return self.net(x)


class MNISTCNN(nn.Module):
    def __init__(self, out_d=10, fc_l=[2304, 128], size_img=[1, 28, 28]):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(size_img[0], 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(fc_l[0], fc_l[1])
        self.fc2 = nn.Linear(fc_l[1], out_d)
        self.out_d = out_d
        self.size_img = size_img

    def forward(self, x, context=None):
        b_size = x.shape[0]
        x = self.conv1(x.view(-1, self.size_img[0], self.size_img[1], self.size_img[2]))
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x).view(b_size, -1)
        return x


class SubMNISTCNN(nn.Module):
    def __init__(self, out_d=10, fc_l=[2304, 128], size_img=[1, 28, 28]):
        super(SubMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(size_img[0], 16, 2, 1)
        self.conv2 = nn.Conv2d(16, 16, 2, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(fc_l[0], fc_l[1])
        self.fc2 = nn.Linear(fc_l[1], out_d)
        self.out_d = out_d
        self.size_img = size_img

    def forward(self, x, context=None):
        b_size = x.shape[0]
        x = self.conv1(x.view(-1, self.size_img[0], self.size_img[1], self.size_img[2]))
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x).view(b_size, -1)
        return x


class CIFAR10CNN(nn.Module):
    def __init__(self, out_d=10, fc_l=[400, 128, 84], size_img=[3, 32, 32], k_size=5):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(size_img[0], 6, k_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, k_size)
        self.fc1 = nn.Linear(fc_l[0], fc_l[1])
        self.fc2 = nn.Linear(fc_l[1], fc_l[2])
        self.fc3 = nn.Linear(fc_l[2], out_d)

        self.out_d = out_d
        self.size_img = size_img

    def forward(self, x, context=None):
        b_size = x.shape[0]
        x = self.pool(F.relu(self.conv1(x.view(-1, self.size_img[0], self.size_img[1], self.size_img[2]))))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(b_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).view(b_size, -1)
        return x


class IdentityNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, context=None):
        return x
