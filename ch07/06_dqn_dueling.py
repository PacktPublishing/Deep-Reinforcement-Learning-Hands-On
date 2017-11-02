#!/usr/bin/env python3
import sys
import gym
import ptan
import time
import numpy as np
import argparse

import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model, common

PONG_MODE = True

if PONG_MODE:
    DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
    MEAN_REWARD_BOUND = 19.5
    RUN_NAME = "pong"
    REPLAY_SIZE = 10000
    SYNC_TARGET_FRAMES = 1000
    REPLAY_START_SIZE = 10000
    EPSILON_DECAY_LAST_FRAME = 10 ** 5
    EPSILON_FINAL = 0.02
    LEARNING_RATE = 0.0001
else:
    DEFAULT_ENV_NAME = "BreakoutNoFrameskip-v4"
    MEAN_REWARD_BOUND = 500
    RUN_NAME = "breakout"
    REPLAY_SIZE = 1000000
    REPLAY_START_SIZE = 50000
    SYNC_TARGET_FRAMES = 10000
    EPSILON_DECAY_LAST_FRAME = 10**6
    EPSILON_FINAL = 0.1
    LEARNING_RATE = 0.00025

GAMMA = 0.99
BATCH_SIZE = 32
EPSILON_START = 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()

    env = gym.make(DEFAULT_ENV_NAME)
    env = ptan.common.wrappers.wrap_dqn(env)

    writer = SummaryWriter(comment="-" + RUN_NAME + "-dueling")
    net = dqn_model.DuelingDQN(env.observation_space.shape, env.action_space.n)
    if args.cuda:
        net.cuda()

    tgt_net = ptan.agent.TargetNet(net)
    epsilon_greedy_selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0)
    agent = ptan.agent.DQNAgent(net, epsilon_greedy_selector, cuda=args.cuda)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
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
        batch = buffer.sample(BATCH_SIZE)
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=GAMMA, cuda=args.cuda)
        loss_v.backward()
        optimizer.step()

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.sync()
    writer.close()
