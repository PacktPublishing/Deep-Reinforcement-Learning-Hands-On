import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

ROLLOUT_HIDDEN = 256


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

        self.reward_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        rw_conv_out = self._get_reward_conv_out((n_planes, ) + input_shape[1:])
        self.reward_fc = nn.Sequential(
            nn.Linear(rw_conv_out, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _get_reward_conv_out(self, shape):
        o = self.conv1(Variable(torch.zeros(1, *shape)))
        o = self.reward_conv(o)
        return int(np.prod(o.size()))

    def forward(self, imgs, actions):
        batch_size = actions.size()[0]
        act_planes_v = Variable(torch.FloatTensor(batch_size, self.n_actions, *self.input_shape[1:]).zero_())
        if actions.is_cuda:
            act_planes_v = act_planes_v.cuda()
        act_planes_v[range(batch_size), actions] = 1.0
        comb_input_v = torch.cat((imgs, act_planes_v), dim=1)
        c1_out = self.conv1(comb_input_v)
        c2_out = self.conv2(c1_out)
        c2_out += c1_out
        img_out = self.deconv(c2_out)
        rew_conv = self.reward_conv(c2_out).view(batch_size, -1)
        rew_out = self.reward_fc(rew_conv)
        return img_out, rew_out


class RolloutEncoder(nn.Module):
    def __init__(self, input_shape, hidden_size=ROLLOUT_HIDDEN):
        super(RolloutEncoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.rnn = nn.LSTM(input_size=conv_out_size+1, hidden_size=hidden_size, batch_first=False)

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, obs_v, reward_v):
        """
        Input is in (time, batch, *) order
        """
        n_time = obs_v.size()[0]
        n_batch = obs_v.size()[1]
        n_items = n_time * n_batch
        obs_flat_v = obs_v.view(n_items, *obs_v.size()[2:])
        conv_out = self.conv(obs_flat_v)
        conv_out = conv_out.view(n_time, n_batch, -1)
        rnn_in = torch.cat((conv_out, reward_v), dim=2)
        _, (rnn_hid, _) = self.rnn(rnn_in)
        return rnn_hid.view(-1)


class I2A(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(I2A, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        fc_input = conv_out_size + ROLLOUT_HIDDEN * n_actions

        self.fc = nn.Sequential(
            nn.Linear(fc_input, 512),
            nn.ReLU()
        )
        self.policy = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x, enc_rollout):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        fc_in = torch.cat((conv_out, enc_rollout), dim=1)
        fc_out = self.fc(fc_in)
        return self.policy(fc_out), self.value(fc_out)
