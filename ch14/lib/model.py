import ptan
import numpy as np

import torch
import torch.nn as nn

HID_SIZE = 128


class ModelA2C(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelA2C, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus(),
        )
        self.value = nn.Linear(HID_SIZE, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)


class ContAgent(ptan.agent.BaseAgent):
    def __init__(self, net, cuda=False, apply_tanh=True):
        self.net = net
        self.cuda = cuda
        self.apply_tanh = apply_tanh

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states, cuda=self.cuda)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        if self.apply_tanh:
            actions = np.tanh(actions)
        return actions, agent_states
