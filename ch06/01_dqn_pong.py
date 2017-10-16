#!/usr/bin/env python3
import gym
import gym.spaces
import numpy as np
import collections
from scipy.misc import imresize

import torch
import torch.nn as nn
from torch.autograd import Variable


class ImageWrapper(gym.ObservationWrapper):
    X_OFS = 20
    def __init__(self, env):
        super(ImageWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(0, 1, self._observation(env.observation_space.low).shape)

    def _observation(self, obs):
        obs = imresize(obs, (110, 84))
        obs = obs.mean(axis=-1, keepdims=True)

        obs = obs[self.X_OFS:self.X_OFS+84, :, :]
        obs = np.moveaxis(obs, 2, 0)
        return obs.astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0))

    def _reset(self):
        self.buffer = self.observation_space.low.copy()
        return self._observation(self.env.reset())

    def _observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class EpsilonGreedyWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=1.0):
        super(EpsilonGreedyWrapper, self).__init__(env)
        self.epsilon = epsilon

    def _action(self, action):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return action

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque()

    def append(self, experience):
        self.buffer.append(experience)
        while len(self.buffer) > self.capacity:
            self.buffer.popleft()

    def sample(self, batch_size):
        return np.random.choice(self.buffer, batch_size, replace=False)


if __name__ == "__main__":
    env = BufferWrapper(ImageWrapper(gym.make("Pong-v4")), n_steps=4)
    env = EpsilonGreedyWrapper(env, epsilon=1.0)
    net = DQN(env.observation_space.shape, env.action_space.n)
    print(net)
    out = net(Variable(torch.FloatTensor([env.reset()])))
    print(out.size())
    pass
