#!/usr/bin/env python3
import gym
import ptan
import time
import numpy as np
import argparse
import collections
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.autograd import Variable

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))


TrainEntry = collections.namedtuple('TrainEntry', field_names=['state', 'adv', 'q', 'action'])


def sum_reward(exps):
    Q = 0
    for idx, e in enumerate(exps):
        Q += (GAMMA ** idx) * e.reward
    return Q


def data_func(net, cuda, train_queue):
    values = collections.deque()
    exp_queue = collections.deque(maxlen=REWARD_STEPS)

    def policy_fun(x):
        policy_v, value_v = net(x)
        values.append(value_v.data.cpu().numpy()[0][0])
        return policy_v

    env = make_env()
    agent = ptan.agent.PolicyAgent(policy_fun, apply_softmax=True, cuda=cuda)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)

    for exp in exp_source:
        exp = exp[0]
        if exp.done:
            print("Done!")
            while exp_queue:
                value_1 = values.popleft()
                Q = sum_reward(exp_queue)
                last_exp = exp_queue.popleft()
                entry = TrainEntry(state=last_exp.state, adv=Q - value_1,
                                   q=Q, action=last_exp.action)
                train_queue.put(entry)
            values.clear()
            continue
        if len(exp_queue) == REWARD_STEPS:
            value_1 = values.popleft()
            value_n = values[-1]
            Q = sum_reward(exp_queue)
            last_exp = exp_queue.popleft()
            Q += GAMMA ** REWARD_STEPS * value_n
            entry = TrainEntry(state=last_exp.state, adv=Q - value_1,
                               q=Q, action=last_exp.action)
            train_queue.put(entry)
        exp_queue.append(exp)

    train_queue.put(None)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    env = make_env()
    net = common.AtariA2C(env.observation_space.shape, env.action_space.n)
    if args.cuda:
        net.cuda()

    train_queue = mp.Queue(maxsize=10)
    data_proc = mp.Process(target=data_func, args=(net, args.cuda, train_queue))
    data_proc.start()

    batch_states = []
    batch_advs = []
    batch_qs = []
    batch_actions = []
    while True:
        train_entry = train_queue.get()
        if train_entry is None:
            data_proc.join()
            break
        batch_states.append(train_entry.state)
        batch_advs.append(train_entry.adv)
        batch_qs.append(train_entry.q)
        batch_actions.append(train_entry.action)

        if len(batch_states) < BATCH_SIZE:
            continue

        print("Train")

        batch_states.clear()
        batch_advs.clear()
        batch_qs.clear()
        batch_actions.clear()

pass
