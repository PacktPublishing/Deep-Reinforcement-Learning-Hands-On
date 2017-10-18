#!/usr/bin/env python3
import gym
import ptan
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)



if __name__ == "__main__":
    env = gym.make("Pong-v4")
    env = ptan.common.wrappers.PreprocessImage(env, height=84, width=84, grayscale=True)
    env = ptan.common.wrappers.FrameBuffer(env, n_frames=4)

    net = DQN(env.observation_space.shape, env.action_space.n)
    epsilon_greedy_selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0)
    agent = ptan.agent.DQNAgent(net, epsilon_greedy_selector)

    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=2)

    print(net)
    o = env.reset()
    n = net(Variable(torch.FloatTensor([o])))
    print(o.shape)
    print(n.size())

    for e in exp_source:
        print(e)
        break
