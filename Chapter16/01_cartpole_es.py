#!/usr/bin/env python3
import gym
import time
import numpy as np

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter


MAX_BATCH_EPISODES = 100
MAX_BATCH_STEPS = 10000
NOISE_STD = 0.01
LEARNING_RATE = 0.001


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
    steps = 0
    while True:
        obs_v = torch.FloatTensor([obs])
        act_prob = net(obs_v)
        acts = act_prob.max(dim=1)[1]
        obs, r, done, _ = env.step(acts.data.numpy()[0])
        reward += r
        steps += 1
        if done:
            break
    return reward, steps


def sample_noise(net):
    pos = []
    neg = []
    for p in net.parameters():
        noise_t = torch.from_numpy(np.random.normal(size=p.data.size()).astype(np.float32))
        pos.append(noise_t)
        neg.append(-noise_t)
    return pos, neg


def eval_with_noise(env, net, noise):
    old_params = net.state_dict()
    for p, p_n in zip(net.parameters(), noise):
        p.data += NOISE_STD * p_n
    r, s = evaluate(env, net)
    net.load_state_dict(old_params)
    return r, s


def train_step(net, batch_noise, batch_reward, writer, step_idx):
    weighted_noise = None
    norm_reward = np.array(batch_reward)
    norm_reward -= np.mean(norm_reward)
    s = np.std(norm_reward)
    if abs(s) > 1e-6:
        norm_reward /= s

    for noise, reward in zip(batch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n
    m_updates = []
    for p, p_update in zip(net.parameters(), weighted_noise):
        update = p_update / (len(batch_reward) * NOISE_STD)
        p.data += LEARNING_RATE * update
        m_updates.append(torch.norm(update))
    writer.add_scalar("update_l2", np.mean(m_updates), step_idx)


if __name__ == "__main__":
    writer = SummaryWriter(comment="-cartpole-es")
    env = gym.make("CartPole-v0")

    net = Net(env.observation_space.shape[0], env.action_space.n)
    print(net)

    step_idx = 0
    while True:
        t_start = time.time()
        batch_noise = []
        batch_reward = []
        batch_steps = 0
        for _ in range(MAX_BATCH_EPISODES):
            noise, neg_noise = sample_noise(net)
            batch_noise.append(noise)
            batch_noise.append(neg_noise)
            reward, steps = eval_with_noise(env, net, noise)
            batch_reward.append(reward)
            batch_steps += steps
            reward, steps = eval_with_noise(env, net, neg_noise)
            batch_reward.append(reward)
            batch_steps += steps
            if batch_steps > MAX_BATCH_STEPS:
                break

        step_idx += 1
        m_reward = np.mean(batch_reward)
        if m_reward > 199:
            print("Solved in %d steps" % step_idx)
            break

        train_step(net, batch_noise, batch_reward, writer, step_idx)
        writer.add_scalar("reward_mean", m_reward, step_idx)
        writer.add_scalar("reward_std", np.std(batch_reward), step_idx)
        writer.add_scalar("reward_max", np.max(batch_reward), step_idx)
        writer.add_scalar("batch_episodes", len(batch_reward), step_idx)
        writer.add_scalar("batch_steps", batch_steps, step_idx)
        speed = batch_steps / (time.time() - t_start)
        writer.add_scalar("speed", speed, step_idx)
        print("%d: reward=%.2f, speed=%.2f f/s" % (step_idx, m_reward, speed))

    pass
