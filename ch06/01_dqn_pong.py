#!/usr/bin/env python3
import argparse
import gym
import gym.spaces
import numpy as np
import collections
from scipy.misc import imresize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

GAMMA = 0.99


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
        self.buffer = np.zeros_like(self.observation_space.low)
        return self._observation(self.env.reset())

    def _observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


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
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset().copy()
        self.total_reward = 0.0

    def play_n_steps(self, steps, net, epsilon=0.0, cuda=False):
        rewards = []
        for _ in range(steps):
            # decide on action
            state_v = Variable(torch.FloatTensor([self.state]))
            if cuda:
                state_v = state_v.cuda()
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = act_v.data.cpu().numpy()[0]
            # apply epsilon greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            # do step in the environment
            new_state, reward, is_done, _ = env.step(action)
            self.total_reward += reward
            new_state = new_state.copy()

            self.exp_buffer.append(Experience(self.state, action, reward, is_done, new_state))
            self.state = new_state
            if is_done:
                rewards.append(self.total_reward)
                self._reset()
        return rewards


def calc_loss(batch, net, cuda=False):
    x = [exp.state for exp in batch]
    x_v = Variable(torch.FloatTensor(x))
    if cuda:
        x_v = x_v.cuda()
    q_v = net(x_v)
    y = q_v.data.cpu().numpy().copy()

    new_x = [exp.new_state for exp in batch]
    new_x_v = Variable(torch.FloatTensor(new_x))
    if cuda:
        new_x_v = new_x_v.cuda()
    new_q_v = net(new_x_v)
    new_q = new_q_v.data.cpu().numpy()

    for idx, exp in enumerate(batch):
        R = exp.reward
        if not exp.done:
            R += GAMMA * np.max(new_q[idx])
        y[idx][exp.action] = R

    y_v = Variable(torch.FloatTensor(y))
    if cuda:
        y_v = y_v.cuda()
    return nn.MSELoss()(q_v, y_v)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable cuda mode")
    args = parser.parse_args()

    writer = SummaryWriter(comment='-pong')
    env = BufferWrapper(ImageWrapper(gym.make("Pong-v4")), n_steps=4)
    net = DQN(env.observation_space.shape, env.action_space.n)
    print(net)

    exp_buffer = ExperienceBuffer(capacity=1000)
    agent = Agent(env, exp_buffer)
    epsilon = 1.0

    optimizer = optim.RMSprop(net.parameters(), lr=0.001)
    if args.cuda:
        net.cuda()

    iter_idx = 0
    while True:
        rewards = agent.play_n_steps(100, net, epsilon=epsilon, cuda=args.cuda)
        losses = []
        for _ in range(4):
            batch = exp_buffer.sample(64)
            optimizer.zero_grad()
            loss_v = calc_loss(batch, net, cuda=args.cuda)
            loss_v.backward()
            losses.append(loss_v.data.cpu().numpy())
        epsilon *= 0.99
        epsilon = max(epsilon, 0.1)
        writer.add_scalar("epsilon", epsilon, iter_idx)
        writer.add_scalar("loss", np.mean(losses), iter_idx)
        if rewards:
            print("%d: rewards: %s" % (iter_idx, rewards))
            writer.add_scalar("reward", np.mean(rewards), iter_idx)
        iter_idx += 1
