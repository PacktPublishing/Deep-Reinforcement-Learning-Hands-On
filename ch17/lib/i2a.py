import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable


class ImagPolicy(nn.Module):
    """
    Imagination policy net, trained with a cross-entropy between the main model-free policy net
    """
    def __init__(self, input_shape, n_actions):
        super(ImagPolicy, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out)


class EnvironmentModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(EnvironmentModel, self).__init__()

        self.input_shape = input_shape
        self.n_actions = n_actions

        # input color planes will be equal to frames plus one-hot encoded actions
        n_planes = input_shape[0] + n_actions
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_planes, 32, kernel_size=4, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.deconv = nn.ConvTranspose2d(32, input_shape[0], kernel_size=4, stride=4, padding=0)

    def forward(self, imgs, actions):
        batch_size = actions.size()[0]
        act_planes_v = Variable(torch.ByteTensor(batch_size, self.n_actions, *self.input_shape[1:]).zero_())
        if actions.is_cuda:
            act_planes_v = act_planes_v.cuda()
        act_planes_v[range(batch_size), actions] = 255
        comb_input_v = torch.cat((imgs, act_planes_v), dim=1)
        c1_out = self.conv1(comb_input_v.float() / 255)
        c2_out = self.conv2(c1_out)
        c2_out += c1_out
        img_out = self.deconv(c2_out)
        return img_out

