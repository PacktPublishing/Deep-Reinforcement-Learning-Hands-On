#!/usr/bin/env python3
import gym
import ptan
import argparse
import numpy as np

import torch.optim as optim

from lib import environ, data, models, common

from tensorboardX import SummaryWriter

BATCH_SIZE = 128
BARS_COUNT = 10
TARGET_NET_SYNC = 1000
DEFAULT_STOCKS = "data/YNDX_160101_161231.csv"

GAMMA = 0.99

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_STEPS = 100000

REWARD_STEPS = 2

LEARNING_RATE = 0.0001

STATES_TO_EVALUATE = 1000
EVAL_EVERY_STEP = 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--data", default=DEFAULT_STOCKS, help="Stocks file to train on, default=" + DEFAULT_STOCKS)
    args = parser.parse_args()

    stock_data = {"YNDX": data.load_relative(args.data)}
    env = environ.StocksEnv(stock_data, bars_count=BARS_COUNT, reset_on_close=True)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    writer = SummaryWriter(comment="-simple")
    net = models.SimpleFFDQN(env.observation_space.shape[0], env.action_space.n)
    if args.cuda:
        net.cuda()
    tgt_net = ptan.agent.TargetNet(net)
    agent = ptan.agent.DQNAgent(net, ptan.actions.ArgmaxActionSelector(), cuda=args.cuda)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.PrioritizedReplayBuffer(exp_source, REPLAY_SIZE, PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    step_idx = 0
    beta = BETA_START
    eval_states = None

    with common.RewardTracker(writer, np.inf, group_rewards=10) as reward_tracker:
        while True:
            step_idx += 1
            buffer.populate(1)
            beta = min(1.0, BETA_START + step_idx * (1.0 - BETA_START) / BETA_STEPS)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                reward_tracker.reward(new_rewards[0], step_idx)

            if len(buffer) < REPLAY_INITIAL:
                continue

            if eval_states is None:
                print("Initial buffer populated, start training")
                eval_states = buffer.sample(STATES_TO_EVALUATE, beta)[0]
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)

            if step_idx % EVAL_EVERY_STEP == 0:
                mean_val = common.calc_values_of_states(eval_states, net, cuda=args.cuda)
                writer.add_scalar("values_mean", mean_val, step_idx)
                writer.add_scalar("beta", beta, step_idx)

            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(BATCH_SIZE, beta)
            loss_v, sample_prios_v = common.calc_loss(batch, batch_weights, net, tgt_net.target_model,
                                                      GAMMA ** REWARD_STEPS, cuda=args.cuda)
            loss_v.backward()
            optimizer.step()
            buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

            if step_idx % TARGET_NET_SYNC == 0:
                tgt_net.sync()

    pass
