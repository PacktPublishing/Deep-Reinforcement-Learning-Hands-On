#!/usr/bin/env python3
import gym
import ptan
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from lib import dqn_model


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


def unpack_batch(batch):
    states = [exp[0].state for exp in batch]
    next_states = [exp[-1].state for exp in batch]
    actions = [exp[0].action for exp in batch]
    rewards = [exp[0].reward for exp in batch]
    dones = [exp[0].done for exp in batch]
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(next_states, copy=False)


def calc_loss(batch, net, tgt_net, cuda=False):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = Variable(torch.from_numpy(states))
    next_states_v = Variable(torch.from_numpy(next_states), volatile=True)
    actions_v = Variable(torch.from_numpy(actions))
    rewards_v = Variable(torch.from_numpy(rewards))
    done_mask = torch.ByteTensor(dones)
    if cuda:
        states_v = states_v.cuda()
        next_states_v = next_states_v.cuda()
        actions_v = actions_v.cuda()
        rewards_v = rewards_v.cuda()
        done_mask = done_mask.cuda()

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values.volatile = False

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()

    env = gym.make("PongNoFrameskip-v4")
    env = ptan.common.wrappers.wrap_dqn(env)
    env = ptan.common.wrappers.ScaledFloatFrame(env)

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    if args.cuda:
        net.cuda()

    tgt_net = ptan.agent.TargetNet(net)
    epsilon_greedy_selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0)
    agent = ptan.agent.DQNAgent(net, epsilon_greedy_selector, cuda=args.cuda)

    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=2)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    total_rewards = []
    while True:
        frame_idx += 1
        buffer.populate(1)
        epsilon_greedy_selector.epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            total_rewards.extend(new_rewards)
            print("%d: %s" % (frame_idx, new_rewards))

        if len(buffer) < REPLAY_START_SIZE:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_v = calc_loss(batch, net, tgt_net.target_model, cuda=args.cuda)
        loss_v.backward()
        optimizer.step()

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.sync()
