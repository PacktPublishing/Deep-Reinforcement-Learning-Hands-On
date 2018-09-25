#!/usr/bin/env python3
import argparse
import numpy as np

from lib import environ, data, models

import torch

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


EPSILON = 0.02


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, help="CSV file with quotes to run the model")
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-b", "--bars", type=int, default=50, help="Count of bars to feed into the model")
    parser.add_argument("-n", "--name", required=True, help="Name to use in output images")
    parser.add_argument("--commission", type=float, default=0.1, help="Commission size in percent, default=0.1")
    parser.add_argument("--conv", default=False, action="store_true", help="Use convolution model instead of FF")
    args = parser.parse_args()

    prices = data.load_relative(args.data)
    env = environ.StocksEnv({"TEST": prices}, bars_count=args.bars, reset_on_close=False, commission=args.commission,
                            state_1d=args.conv, random_ofs_on_reset=False, reward_on_close=False, volumes=False)
    if args.conv:
        net = models.DQNConv1D(env.observation_space.shape, env.action_space.n)
    else:
        net = models.SimpleFFDQN(env.observation_space.shape[0], env.action_space.n)

    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    obs = env.reset()
    start_price = env._state._cur_close()

    total_reward = 0.0
    step_idx = 0
    rewards = []

    while True:
        step_idx += 1
        obs_v = torch.tensor([obs])
        out_v = net(obs_v)
        action_idx = out_v.max(dim=1)[1].item()
        if np.random.random() < EPSILON:
            action_idx = env.action_space.sample()
        action = environ.Actions(action_idx)

        obs, reward, done, _ = env.step(action_idx)
        total_reward += reward
        rewards.append(total_reward)
        if step_idx % 100 == 0:
            print("%d: reward=%.3f" % (step_idx, total_reward))
        if done:
            break

    plt.clf()
    plt.plot(rewards)
    plt.title("Total reward, data=%s" % args.name)
    plt.ylabel("Reward, %")
    plt.savefig("rewards-%s.png" % args.name)
