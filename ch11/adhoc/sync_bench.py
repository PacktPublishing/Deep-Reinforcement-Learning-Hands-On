#!/usr/bin/env python3
import time
import timeit

import gym
import ptan
import os
import sys
sys.path.append(os.getcwd())
from lib import common

import torch
import torch.nn as nn


# Results:
# Original sync, number=100, cuda=True, speed=7634.508 runs/s
# Original sync, number=1000, cuda=True, speed=8606.037 runs/s
# Original sync, number=10000, cuda=True, speed=8822.823 runs/s
# Original sync, number=100000, cuda=True, speed=8842.458 runs/s
#
# Original sync, number=100, cuda=False, speed=779.575 runs/s
# Original sync, number=1000, cuda=False, speed=767.816 runs/s
# Original sync, number=10000, cuda=False, speed=770.027 runs/s
# Original sync, number=100000, cuda=False, speed=755.772 runs/s

# New sync, async=False
# New sync, number=100, cuda=True, speed=6001.022 runs/s
# New sync, number=1000, cuda=True, speed=6087.863 runs/s
# New sync, number=10000, cuda=True, speed=6083.333 runs/s
# New sync, number=100000, cuda=True, speed=6096.957 runs/s

# async=True
# New sync, number=100, cuda=True, speed=5574.816 runs/s
# New sync, number=1000, cuda=True, speed=6006.258 runs/s
# New sync, number=10000, cuda=True, speed=6053.777 runs/s
# New sync, number=100000, cuda=True, speed=6074.822 runs/s



CUDA = True
REPEAT_NUMBER = 100


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))


def new_sync(tgt_net, src_net):
    assert isinstance(tgt_net, nn.Module)
    assert isinstance(src_net, nn.Module)
    for tgt, src in zip(tgt_net.parameters(), src_net.parameters()):
        tgt.data.copy_(src.data, broadcast=False, async=True)


if __name__ == "__main__":
    env = make_env()
    net = common.AtariA2C(env.observation_space.shape, env.action_space.n)
    if CUDA:
        net.cuda()
        print("Initial sleep 20 seconds")
        time.sleep(20)

    tgt_net = ptan.agent.TargetNet(net)
    ns = globals()
    ns.update(locals())
    for number in [100, 1000, 10000, 100000]:
        t = timeit.timeit('tgt_net.sync()', number=number, globals=ns)
        print("Original sync, number=%d, cuda=%s, speed=%.3f runs/s" % (number, CUDA, number / t))

    for number in [100, 1000, 10000, 100000]:
        t = timeit.timeit('new_sync(tgt_net.target_model, net)', number=number, globals=ns)
        print("New sync, number=%d, cuda=%s, speed=%.3f runs/s" % (number, CUDA, number / t))
