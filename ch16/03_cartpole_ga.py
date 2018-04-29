#!/usr/bin/env python3
import gym
import copy
import numpy as np

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

NOISE_STD = 0.01
POPULATION_SIZE = 50
PARENTS_COUNT = 10


class Net(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


def evaluate(env, net):
    obs = env.reset()
    reward = 0.0
    while True:
        obs_v = torch.FloatTensor([obs])
        act_prob = net(obs_v)
        acts = act_prob.max(dim=1)[1]
        obs, r, done, _ = env.step(acts.data.numpy()[0])
        reward += r
        if done:
            break
    return reward


def mutate_parent(net):
    new_net = copy.deepcopy(net)
    for p in new_net.parameters():
        noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32))
        p.data += NOISE_STD * noise_t
    return new_net


if __name__ == "__main__":
    writer = SummaryWriter(comment="-cartpole-ga")
    env = gym.make("CartPole-v0")

    gen_idx = 0
    nets = [
        Net(env.observation_space.shape[0], env.action_space.n)
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
        if reward_mean > 199:
            print("Solved in %d steps" % gen_idx)
            break

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
