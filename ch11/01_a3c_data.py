#!/usr/bin/env python3
import gym
import ptan
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


def play_funct(net, cuda, train_queue):
    values = collections.deque()
    exp_queue = collections.deque(maxlen=REWARD_STEPS)

    def policy_fun(x):
        policy_v, value_v = net(x)
        values.append(value_v.data.cpu().numpy())
        return policy_v

    env = make_env()
    agent = ptan.agent.PolicyAgent(policy_fun, apply_softmax=True, cuda=cuda)
    exp_source = ptan.experience.ExperienceSource(env, agent)

    for exp in exp_source:
        if exp.done:
            while exp_queue:
                value_1 = values.popleft()
                Q = sum_reward(exp_queue)
                last_exp = exp_queue.popleft()
                entry = TrainEntry(state=last_exp.state, adv=Q - value_1,
                                   q=Q, action=last_exp.action)
                train_queue.put(entry)
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

pass
