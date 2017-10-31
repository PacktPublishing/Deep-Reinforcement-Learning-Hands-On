#!/usr/bin/env python3
import sys
import gym
import ptan
import time
import numpy as np
import argparse
import collections

import torch
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from lib import dqn_model

PONG_MODE = False

if PONG_MODE:
    DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
    MEAN_REWARD_BOUND = 19.5
    RUN_NAME = "pong"
else:
    DEFAULT_ENV_NAME = "SpaceInvadersNoFrameskip-v4"
    MEAN_REWARD_BOUND = 1000
    RUN_NAME = "invaders"

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

PRIO_REPLAY_ALPHA = 0.6
PRIO_REPLAY_BETA = 0.4


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_loss(batch, batch_weights, net, tgt_net, cuda=False):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = Variable(torch.from_numpy(states))
    next_states_v = Variable(torch.from_numpy(next_states), volatile=True)
    actions_v = Variable(torch.from_numpy(actions))
    rewards_v = Variable(torch.from_numpy(rewards))
    done_mask = torch.ByteTensor(dones)
    batch_weights_v = Variable(torch.from_numpy(batch_weights))
    if cuda:
        states_v = states_v.cuda()
        next_states_v = next_states_v.cuda()
        actions_v = actions_v.cuda()
        rewards_v = rewards_v.cuda()
        done_mask = done_mask.cuda()
        batch_weights_v = batch_weights_v.cuda()

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values.volatile = False

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    losses_v = batch_weights_v * (state_action_values - expected_state_action_values) ** 2
    return losses_v.mean(), losses_v


class PrioReplayBuffer:
    def __init__(self, exp_source, buf_size, prob_alpha):
        self.exp_source_iter = iter(exp_source)
        self.prob_alpha = prob_alpha
        self.buffer = collections.deque(maxlen=buf_size)
        self.priorities = collections.deque(maxlen=buf_size)

    def __len__(self):
        return len(self.buffer)

    def populate(self, samples):
        max_prio = max(self.priorities) if self.priorities else 1.0
        for _ in range(samples):
            self.buffer.append(next(self.exp_source_iter))
            self.priorities.append(max_prio)

    def sample(self, batch_size, beta):
        probs = np.array(self.priorities, dtype=np.float32) ** self.prob_alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()

    env = gym.make(DEFAULT_ENV_NAME)
    env = ptan.common.wrappers.wrap_dqn(env)

    writer = SummaryWriter(comment="-" + RUN_NAME + "-prio-replay")
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    if args.cuda:
        net.cuda()

    tgt_net = ptan.agent.TargetNet(net)
    epsilon_greedy_selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0)
    agent = ptan.agent.DQNAgent(net, epsilon_greedy_selector, cuda=args.cuda)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = PrioReplayBuffer(exp_source, REPLAY_SIZE, PRIO_REPLAY_ALPHA)
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
            sys.stdout.flush()
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
        batch, batch_indices, batch_weights = buffer.sample(BATCH_SIZE, PRIO_REPLAY_BETA)
        loss_v, sample_prios_v = calc_loss(batch, batch_weights, net, tgt_net.target_model, cuda=args.cuda)
        loss_v.backward()
        optimizer.step()
        buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.sync()
    writer.close()
