#!/usr/bin/env python3
import argparse
import numpy as np

from lib import environ, data, models

import torch
from torch.autograd import Variable

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, help="CSV file with quotes to run the model")
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-b", "--bars", type=int, default=50, help="Count of bars to feed into the model")
    parser.add_argument("-n", "--name", required=True, help="Name to use in output images")
    parser.add_argument("--conv", default=False, action="store_true", help="Use convolution model instead of FF")
    args = parser.parse_args()

    prices = data.load_relative(args.data)
    env = environ.StocksEnv({"TEST": prices}, bars_count=args.bars, reset_on_close=False,
                            state_1d=args.conv, random_ofs_on_reset=False, reward_on_close=True)
    if args.conv:
        net = models.DQNConv1D(env.observation_space.shape, env.action_space.n)
    else:
        net = models.SimpleFFDQN(env.observation_space.shape[0], env.action_space.n)

    net.load_state_dict(torch.load(args.model))

    obs = env.reset()
    total_reward = 0.0
    step_idx = 0
    rewards = []

    while True:
        step_idx += 1
        obs_v = Variable(torch.from_numpy(np.expand_dims(obs, 0)))
        out_v = net(obs_v)
        action_idx = out_v.max(dim=1)[1].data.cpu().numpy()[0]
        action = environ.Actions(action_idx)
        obs, reward, done, _ = env.step(action_idx)
#        print(action, reward)
        total_reward += reward
        rewards.append(total_reward)
        if step_idx % 100 == 0:
            print("%d: %.3f" % (step_idx, total_reward))
        if done:
            break
    print(total_reward)

    plt.clf()
    plt.plot(rewards)
    plt.savefig("rewards-%s.png" % args.name)

    pass
