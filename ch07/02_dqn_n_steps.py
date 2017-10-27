#!/usr/bin/env python3
import gym
import ptan
import time
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from lib import dqn_model


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

REWARD_STEPS = 2
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.1


def unpack_batch(batch):
    states = [exp[0].state for exp in batch]
    last_states = [exp[-1].state for exp in batch]
    actions = [exp[0].action for exp in batch]
    rewards = []
    dones = []
    for exp in batch:
        r = 0.0
        was_done = False
        for e in reversed(exp[:-1]):
            r *= GAMMA
            r += e.reward
            was_done |= e.done
        rewards.append(r)
        dones.append(was_done)
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_loss(batch, net, tgt_net, cuda=False):
    states, actions, rewards, dones, last_states = unpack_batch(batch)

    states_v = Variable(torch.from_numpy(states))
    last_states_v = Variable(torch.from_numpy(last_states), volatile=True)
    actions_v = Variable(torch.from_numpy(actions))
    rewards_v = Variable(torch.from_numpy(rewards))
    done_mask = torch.ByteTensor(dones)
    if cuda:
        states_v = states_v.cuda()
        last_states_v = last_states_v.cuda()
        actions_v = actions_v.cuda()
        rewards_v = rewards_v.cuda()
        done_mask = done_mask.cuda()

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    last_state_values = tgt_net(last_states_v).max(1)[0]
    last_state_values[done_mask] = 0.0
    last_state_values.volatile = False

    expected_state_action_values = last_state_values * (GAMMA**REWARD_STEPS) + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()

    env = gym.make(DEFAULT_ENV_NAME)
    env = ptan.common.wrappers.wrap_dqn(env)
    env = ptan.common.wrappers.ScaledFloatFrame(env)

    writer = SummaryWriter(comment="-breakout-2-steps")
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    if args.cuda:
        net.cuda()

    tgt_net = ptan.agent.TargetNet(net)
    epsilon_greedy_selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0)
    agent = ptan.agent.DQNAgent(net, epsilon_greedy_selector, cuda=args.cuda)

    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=REWARD_STEPS+1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    ts_frame = 0
    ts = time.time()

    total_rewards = []
    while True:
        frame_idx += 1
        buffer.populate(1)
        epsilon_greedy_selector.epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            total_rewards.extend(new_rewards)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon_greedy_selector.epsilon,
                speed
            ))
            writer.add_scalar("epsilon", epsilon_greedy_selector.epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", new_rewards[0], frame_idx)
            if mean_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_v = calc_loss(batch, net, tgt_net.target_model, cuda=args.cuda)
        loss_v.backward()
        optimizer.step()

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.sync()
