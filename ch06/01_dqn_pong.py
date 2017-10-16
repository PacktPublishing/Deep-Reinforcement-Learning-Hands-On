#!/usr/bin/env python3
import gym
import gym.spaces
import numpy as np
from scipy.misc import imresize

import torch
import torchvision.utils as tv_utils


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
        print("Reset")
        self.buffer = self.observation_space.low.copy()
        return self._observation(self.env.reset())

    def _observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


if __name__ == "__main__":
    env = BufferWrapper(ImageWrapper(gym.make("Pong-v4")), n_steps=4)
    print(env.observation_space)
    print(env.action_space)
    pass
