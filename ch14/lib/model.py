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


class ModelDDPG(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelDDPG, self).__init__()

        self.n_actor = nn.Sequential(
            nn.Linear(obs_size, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, act_size),
            nn.Tanh()
        )

        self.n_critic = nn.Sequential(
            nn.Linear(obs_size + act_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def actor(self, x):
        return self.n_actor(x)

    def critic(self, obs, act):
        critic_input = torch.cat((obs, act), dim=1)
        return self.n_critic(critic_input)

    def forward(self, x):
        action = self.actor(x)
        return action, self.critic(x, action)


class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, cuda=False):
        self.net = net
        self.cuda = cuda

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states, cuda=self.cuda)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states


class AgentDDPG(ptan.agent.BaseAgent):
    def __init__(self, net, cuda=False):
        self.net = net
        self.cuda = cuda

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states, cuda=self.cuda)
        mu_v = self.net.actor(states_v)
        actions = mu_v.data.cpu().numpy()
        actions = np.clip(actions, -1, 1)
        return actions, agent_states

pass
