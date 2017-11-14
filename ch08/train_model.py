#!/usr/bin/env python3
import os
import gym
import ptan
import argparse
import numpy as np

import torch
import torch.optim as optim

from lib import environ, data, models, common

from tensorboardX import SummaryWriter

BATCH_SIZE = 32
BARS_COUNT = 10
TARGET_NET_SYNC = 1000
DEFAULT_STOCKS = "data/YNDX_160101_161231.csv"

GAMMA = 0.99

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000

REWARD_STEPS = 2

LEARNING_RATE = 0.0001

STATES_TO_EVALUATE = 1000
EVAL_EVERY_STEP = 1000

EPSILON_START = 1.0
EPSILON_STOP = 0.1
EPSILON_STEPS = 1000000

PRERTRAIN_ITERATIONS = 100000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--data", default=DEFAULT_STOCKS, help="Stocks file or dir to train on, default=" + DEFAULT_STOCKS)
    parser.add_argument("-r", "--run", required=True, help="Run name")
    args = parser.parse_args()

    saves_path = os.path.join("saves", args.run)
    os.makedirs(saves_path, exist_ok=True)

    if os.path.isfile(args.data):
        stock_data = {"YNDX": data.load_relative(args.data)}
        env = environ.StocksEnv(stock_data, bars_count=BARS_COUNT, reset_on_close=True)
    elif os.path.isdir(args.data):
        env = environ.StocksEnv.from_dir(args.data, bars_count=BARS_COUNT, reset_on_close=True)
    stocks_env = env
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    writer = SummaryWriter(comment="-simple-" + args.run)
    net = models.SimpleFFDQN(env.observation_space.shape[0], env.action_space.n)
    if args.cuda:
        net.cuda()
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector, cuda=args.cuda)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # main training loop
    step_idx = 0
    eval_states = None
    max_reward = None
    best_mean_val = None

    with common.RewardTracker(writer, np.inf, group_rewards=10) as reward_tracker:
        while True:
            step_idx += 1
            buffer.populate(1)
            selector.epsilon = max(EPSILON_STOP, EPSILON_START - step_idx / EPSILON_STEPS)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                reward_tracker.reward(new_rewards[0], step_idx, selector.epsilon)
                if max_reward is None or max_reward < new_rewards[0]:
                    if max_reward is not None:
                        print("%d: Max reward updated %.3f -> %.3f" % (step_idx, max_reward, new_rewards[0]))
                    max_reward = new_rewards[0]
                    writer.add_scalar("reward_max", max_reward, step_idx)
                    torch.save(net.state_dict(), os.path.join(saves_path, "best-%.3f.data" % max_reward))

            if len(buffer) < REPLAY_INITIAL:
                continue

            if eval_states is None:
                print("Initial buffer populated, start training")
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)

            if step_idx % EVAL_EVERY_STEP == 0:
                mean_val = common.calc_values_of_states(eval_states, net, cuda=args.cuda)
                writer.add_scalar("values_mean", mean_val, step_idx)
                if best_mean_val is None or best_mean_val < mean_val:
                    if best_mean_val is not None:
                        print("%d: Best mean value updated %.3f -> %.3f" % (step_idx, best_mean_val, mean_val))
                    best_mean_val = mean_val
                    torch.save(net.state_dict(), os.path.join(saves_path, "mean_val-%.3f.data" % mean_val))

            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_v = common.calc_loss(batch, net, tgt_net.target_model,
                                                      GAMMA ** REWARD_STEPS, cuda=args.cuda)
            loss_v.backward()
            optimizer.step()

            if step_idx % TARGET_NET_SYNC == 0:
                tgt_net.sync()

    pass
