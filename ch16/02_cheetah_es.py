#!/usr/bin/env python3
import gym
import roboschool
import ptan
import time
import argparse
import numpy as np
import collections

import torch
import torch.nn as nn
from torch import multiprocessing as mp
from torch import optim

from tensorboardX import SummaryWriter

NOISE_STD = 0.05
LEARNING_RATE = 0.01
PROCESSES_COUNT = 6
ITERS_PER_UPDATE = 10
MAX_ITERS = 100000

# result item from the worker to master. Fields:
# 1. random seed used to generate noise
# 2. reward obtained from the positive noise
# 3. reward obtained from the negative noise
# 4. total amount of steps done
RewardsItem = collections.namedtuple('RewardsItem', field_names=['seed', 'pos_reward', 'neg_reward', 'steps'])


def make_env():
    return gym.make("RoboschoolHalfCheetah-v1")


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


def evaluate(env, net, device="cpu"):
    obs = env.reset()
    reward = 0.0
    steps = 0
    while True:
        obs_v = ptan.agent.default_states_preprocessor([obs]).to(device)
        action_v = net(obs_v)
        action = action_v.data.cpu().numpy()[0]
        obs, r, done, _ = env.step(action)
        reward += r
        steps += 1
        if done:
            break
    return reward, steps


def sample_noise(net, device="cpu"):
    res = []
    neg = []
    for p in net.parameters():
        noise_t = torch.FloatTensor(np.random.normal(size=p.data.size()).astype(np.float32)).to(device)
        res.append(noise_t)
        neg.append(-noise_t)
    return res, neg


def eval_with_noise(env, net, noise, noise_std, device="cpu"):
    for p, p_n in zip(net.parameters(), noise):
        p.data += noise_std * p_n
    r, s = evaluate(env, net, device)
    for p, p_n in zip(net.parameters(), noise):
        p.data -= noise_std * p_n
    return r, s


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def train_step(optimizer, net, batch_noise, batch_reward, writer, step_idx, noise_std):
    weighted_noise = None
    norm_reward = compute_centered_ranks(np.array(batch_reward))

    for noise, reward in zip(batch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n
    m_updates = []
    optimizer.zero_grad()
    for p, p_update in zip(net.parameters(), weighted_noise):
        update = p_update / (len(batch_reward) * noise_std)
        p.grad = -update
        m_updates.append(torch.norm(update))
    writer.add_scalar("update_l2", np.mean(m_updates), step_idx)
    optimizer.step()


def worker_func(worker_id, params_queue, rewards_queue, device, noise_std):
    env = make_env()
    net = Net(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    net.eval()

    while True:
        params = params_queue.get()
        if params is None:
            break
        net.load_state_dict(params)

        for _ in range(ITERS_PER_UPDATE):
            seed = np.random.randint(low=0, high=65535)
            np.random.seed(seed)
            noise, neg_noise = sample_noise(net, device=device)
            pos_reward, pos_steps = eval_with_noise(env, net, noise, noise_std, device=device)
            neg_reward, neg_steps = eval_with_noise(env, net, neg_noise, noise_std, device=device)
            rewards_queue.put(RewardsItem(seed=seed, pos_reward=pos_reward,
                                          neg_reward=neg_reward, steps=pos_steps+neg_steps))
    pass


if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable CUDA mode")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--noise-std", type=float, default=NOISE_STD)
    parser.add_argument("--iters", type=int, default=MAX_ITERS)
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    writer = SummaryWriter(comment="-cheetah-es_lr=%.3e_sigma=%.3e" % (args.lr, args.noise_std))
    env = make_env()
    net = Net(env.observation_space.shape[0], env.action_space.shape[0])
    print(net)

    params_queues = [mp.Queue(maxsize=1) for _ in range(PROCESSES_COUNT)]
    rewards_queue = mp.Queue(maxsize=ITERS_PER_UPDATE)
    workers = []

    for idx, params_queue in enumerate(params_queues):
        proc = mp.Process(target=worker_func, args=(idx, params_queue, rewards_queue, device, args.noise_std))
        proc.start()
        workers.append(proc)

    print("All started!")
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    for step_idx in range(args.iters):
        # broadcasting network params
        params = net.state_dict()
        for q in params_queues:
            q.put(params)

        # waiting for results
        t_start = time.time()
        batch_noise = []
        batch_reward = []
        results = 0
        batch_steps = 0
        batch_steps_data = []
        while True:
            while not rewards_queue.empty():
                reward = rewards_queue.get_nowait()
                np.random.seed(reward.seed)
                noise, neg_noise = sample_noise(net)
                batch_noise.append(noise)
                batch_reward.append(reward.pos_reward)
                batch_noise.append(neg_noise)
                batch_reward.append(reward.neg_reward)
                results += 1
                batch_steps += reward.steps
                batch_steps_data.append(reward.steps)

            if results == PROCESSES_COUNT * ITERS_PER_UPDATE:
                break
            time.sleep(0.01)

        dt_data = time.time() - t_start
        m_reward = np.mean(batch_reward)
        train_step(optimizer, net, batch_noise, batch_reward, writer, step_idx, args.noise_std)
        writer.add_scalar("reward_mean", m_reward, step_idx)
        writer.add_scalar("reward_std", np.std(batch_reward), step_idx)
        writer.add_scalar("reward_max", np.max(batch_reward), step_idx)
        writer.add_scalar("batch_episodes", len(batch_reward), step_idx)
        writer.add_scalar("batch_steps", batch_steps, step_idx)
        speed = batch_steps / (time.time() - t_start)
        writer.add_scalar("speed", speed, step_idx)
        dt_step = time.time() - t_start - dt_data

        print("%d: reward=%.2f, speed=%.2f f/s, data_gather=%.3f, train=%.3f, steps_mean=%.2f, min=%.2f, max=%.2f, steps_std=%.2f" % (
            step_idx, m_reward, speed, dt_data, dt_step, np.mean(batch_steps_data),
            np.min(batch_steps_data), np.max(batch_steps_data), np.std(batch_steps_data)))

    for worker, p_queue in zip(workers, params_queues):
        p_queue.put(None)
        worker.join()
