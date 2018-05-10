#!/usr/bin/env python3
import sys
import gym
import roboschool
import argparse
import itertools
import collections
import copy
import time
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter


NOISE_STD = 0.005
POPULATION_SIZE = 2000
PARENTS_COUNT = 10
WORKERS_COUNT = 2
SEEDS_PER_WORKER = POPULATION_SIZE // WORKERS_COUNT
MAX_SEED = 2**32 - 1


class MultiNoiseLinear(nn.Linear):
    def set_noise_dim(self, dim):
        assert isinstance(dim, int)
        assert dim > 0
        self.register_buffer('noise', torch.FloatTensor(dim, self.out_features, self.in_features))
        self.register_buffer('noise_bias', torch.FloatTensor(dim, self.out_features))

    def sample_noise_row(self, row):
        # sample noise for our params
        w_noise = NOISE_STD * torch.tensor(np.random.normal(size=self.weight.data.size()).astype(np.float32))
        b_noise = NOISE_STD * torch.tensor(np.random.normal(size=self.bias.data.size()).astype(np.float32))
        self.noise[row].copy_(w_noise)
        self.noise_bias[row].copy_(b_noise)

    def zero_noise(self):
        self.noise.zero_()
        self.noise_bias.zero_()

    def forward(self, x):
        o = super(MultiNoiseLinear, self).forward(x)
        o_n = torch.matmul(self.noise, x.data.unsqueeze(-1)).squeeze(-1)
        o.data += o_n + self.noise_bias
        return o


class Net(nn.Module):
    def __init__(self, obs_size, act_size, hid_size=64):
        super(Net, self).__init__()

        self.nonlin = nn.Tanh()
        self.l1 = MultiNoiseLinear(obs_size, hid_size)
        self.l2 = MultiNoiseLinear(hid_size, hid_size)
        self.l3 = MultiNoiseLinear(hid_size, act_size)

    def forward(self, x):
        l1 = self.nonlin(self.l1(x))
        l2 = self.nonlin(self.l2(l1))
        l3 = self.nonlin(self.l3(l2))
        return l3

    def set_noise_seeds(self, seeds):
        batch_size = len(seeds)
        self.l1.set_noise_dim(batch_size)
        self.l2.set_noise_dim(batch_size)
        self.l3.set_noise_dim(batch_size)

        for idx, seed in enumerate(seeds):
            np.random.seed(seed)
            self.l1.sample_noise_row(idx)
            self.l2.sample_noise_row(idx)
            self.l3.sample_noise_row(idx)

    def zero_noise(self, batch_size):
        self.l1.set_noise_dim(batch_size)
        self.l2.set_noise_dim(batch_size)
        self.l3.set_noise_dim(batch_size)
        self.l1.zero_noise()
        self.l2.zero_noise()
        self.l3.zero_noise()


def evaluate(env, net, device="cpu"):
    obs = env.reset()
    reward = 0.0
    steps = 0
    while True:
        obs_v = torch.FloatTensor([obs]).to(device)
        action_v = net(obs_v)
        obs, r, done, _ = env.step(action_v.data.cpu().numpy()[0])
        reward += r
        steps += 1
        if done:
            break
    return reward, steps


def evaluate_batch(envs, net, device="cpu"):
    count = len(envs)
    obs = [e.reset() for e in envs]
    rewards = [0.0 for _ in range(count)]
    steps = [0 for _ in range(count)]
    done_set = set()

    while len(done_set) < count:
        obs_v = torch.FloatTensor(obs).to(device)
        out_v = net(obs_v)
        out = out_v.data.cpu().numpy()
        for i in range(count):
            if i in done_set:
                continue
            new_o, r, done, _ = envs[i].step(out[i])
            obs[i] = new_o
            rewards[i] += r
            steps[i] += 1
            if done:
                done_set.add(i)
    return rewards, steps


def mutate_net(net, seed, copy_net=True):
    new_net = copy.deepcopy(net) if copy_net else net
    np.random.seed(seed)
    for p in new_net.parameters():
        noise_t = torch.from_numpy(np.random.normal(size=p.data.size()).astype(np.float32))
        p.data += NOISE_STD * noise_t
    return new_net


def build_net(env, seeds):
    torch.manual_seed(seeds[0])
    net = Net(env.observation_space.shape[0], env.action_space.shape[0])
    for seed in seeds[1:]:
        net = mutate_net(net, seed, copy_net=False)
    return net


OutputItem = collections.namedtuple('OutputItem', field_names=['seeds', 'reward', 'steps'])


def worker_func(input_queue, output_queue, device="cpu"):
    env_pool = [gym.make("RoboschoolHalfCheetah-v1")]

    # first generation -- just evaluate given single seeds
    parents = input_queue.get()
    for seed in parents:
        net = build_net(env_pool[0], seed).to(device)
        net.zero_noise(batch_size=1)
        reward, steps = evaluate(env_pool[0], net, device)
        output_queue.put((seed, reward, steps))

    while True:
        parents = input_queue.get()
        if parents is None:
            break
        parents.sort()
        for parent_seeds, children_iter in itertools.groupby(parents, key=lambda s: s[:-1]):
            batch = list(children_iter)
            children_seeds = [b[-1] for b in batch]
            net = build_net(env_pool[0], parent_seeds).to(device)
            net.set_noise_seeds(children_seeds)
            batch_size = len(children_seeds)
            while len(env_pool) < batch_size:
                env_pool.append(gym.make("RoboschoolHalfCheetah-v1"))
            rewards, steps = evaluate_batch(env_pool[:batch_size], net, device)
            for seeds, reward, step in zip(batch, rewards, steps):
                output_queue.put((seeds, reward, step))


if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true')
    args = parser.parse_args()
    writer = SummaryWriter(comment="-cheetah-ga-batch")
    device = "cuda" if args.cuda else "cpu"

    input_queues = []
    output_queue = mp.Queue(maxsize=WORKERS_COUNT)
    workers = []
    for _ in range(WORKERS_COUNT):
        input_queue = mp.Queue(maxsize=1)
        input_queues.append(input_queue)
        w = mp.Process(target=worker_func, args=(input_queue, output_queue, device))
        w.start()
        seeds = [(np.random.randint(MAX_SEED),) for _ in range(SEEDS_PER_WORKER)]
        input_queue.put(seeds)

    gen_idx = 0
    elite = None
    while True:
        t_start = time.time()
        batch_steps = 0
        population = []
        while len(population) < SEEDS_PER_WORKER * WORKERS_COUNT:
            seeds, reward, steps = output_queue.get()
            population.append((seeds, reward))
            batch_steps += steps
        if elite is not None:
            population.append(elite)
        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:PARENTS_COUNT]]
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)
        writer.add_scalar("reward_mean", reward_mean, gen_idx)
        writer.add_scalar("reward_std", reward_std, gen_idx)
        writer.add_scalar("reward_max", reward_max, gen_idx)
        writer.add_scalar("batch_steps", batch_steps, gen_idx)
        writer.add_scalar("gen_seconds", time.time() - t_start, gen_idx)
        speed = batch_steps / (time.time() - t_start)
        writer.add_scalar("speed", speed, gen_idx)
        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s" % (
            gen_idx, reward_mean, reward_max, reward_std, speed))

        elite = population[0]
        for worker_queue in input_queues:
            seeds = []
            for _ in range(SEEDS_PER_WORKER):
                parent = np.random.randint(PARENTS_COUNT)
                next_seed = np.random.randint(MAX_SEED)
                seeds.append(tuple(list(population[parent][0]) + [next_seed]))
            worker_queue.put(seeds)
        gen_idx += 1

    pass
