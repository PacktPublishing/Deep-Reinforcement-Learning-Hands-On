#!/usr/bin/env python3
import gym
import roboschool
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from tensorboardX import SummaryWriter

NOISE_STD = 0.005
POPULATION_SIZE = 2000
PARENTS_COUNT = 200


class Net(nn.Module):
    def __init__(self, obs_size, act_size, hid_size=64):
        super(Net, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, act_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mu(x)


def evaluate(env, net):
    obs = env.reset()
    reward = 0.0
    while True:
        obs_v = Variable(torch.from_numpy(np.array([obs], dtype=np.float32)), volatile=True)
        action_v = net(obs_v)
        obs, r, done, _ = env.step(action_v.data.numpy()[0])
        reward += r
        if done:
            break
    return reward


def mutate_parent(net):
    new_net = copy.deepcopy(net)
    for p in new_net.parameters():
        noise_t = torch.from_numpy(np.random.normal(size=p.data.size()).astype(np.float32))
        p.data += NOISE_STD * noise_t
    return new_net


if __name__ == "__main__":
    writer = SummaryWriter(comment="-cheetah-ga")
    env = gym.make("RoboschoolHalfCheetah-v1")

    gen_idx = 0
    nets = [
        Net(env.observation_space.shape[0], env.action_space.shape[0])
        for _ in range(POPULATION_SIZE)
    ]
    population = [
        (net, evaluate(env, net))
        for net in nets
    ]
    while True:
        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:PARENTS_COUNT]]
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)
        writer.add_scalar("reward_mean", reward_mean, gen_idx)
        writer.add_scalar("reward_std", reward_std, gen_idx)
        writer.add_scalar("reward_max", reward_max, gen_idx)
        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f" % (
            gen_idx, reward_mean, reward_max, reward_std))

        # generate next population
        prev_population = population
        population = [population[0]]
        for _ in range(POPULATION_SIZE-1):
            parent_idx = np.random.randint(0, PARENTS_COUNT)
            parent = prev_population[parent_idx][0]
            net = mutate_parent(parent)
            fitness = evaluate(env, net)
            population.append((net, fitness))
        gen_idx += 1

    pass
