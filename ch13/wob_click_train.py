#!/usr/bin/env python3
import gym
import time
import universe
import argparse
import numpy as np

from lib import wob_vnc, model_vnc

import ptan
import torch
from torch.autograd import Variable

ENV_NAME = "wob.mini.BisectAngle-v0"
#REMOTE_ADDR = 'vnc://gpu:5900+15900'
#REMOTE_ADDR = 4
REMOTE_ADDR = 'vnc://gpu:5900+15900,gpu:5901+15901'

# To start multiple remote containers, use something like this
# docker run -d -p 5900:5900 -p 15900:15900 --privileged --ipc host --cap-add SYS_ADMIN quay.io/openai/universe.world-of-bits:0.20.0
# docker run -d -p 5901:5900 -p 15901:15900 --privileged --ipc host --cap-add SYS_ADMIN quay.io/openai/universe.world-of-bits:0.20.0


GAMMA = 0.99
REWARD_STEPS = 4
BATCH_SIZE = 32


def step_env(env, action):
    while True:
        obs, reward, is_done, info = env.step([action])
        if obs[0] is None:
            continue
        break
    return obs, reward, is_done, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--cuda", default=False, action='store_true', help="CUDA mode")
    args = parser.parse_args()

    env = gym.make(ENV_NAME)
    env = universe.wrappers.experimental.SoftmaxClickMouse(env)
    env = wob_vnc.MiniWoBCropper(env)

    env.configure(remotes=REMOTE_ADDR)

    net = model_vnc.Model(input_shape=(3, wob_vnc.HEIGHT, wob_vnc.WIDTH),
                          n_actions=env.action_space.n)
    print(net)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], cuda=args.cuda,
                                   apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        [env], agent, gamma=GAMMA, steps_count=REWARD_STEPS, vectorized=True)

    # obs, reward, done, info = step_env(env, env.action_space.sample())
    # obs_v = Variable(torch.from_numpy(np.array(obs)))
    # r = net(obs_v)
    # print(r[0].size(), r[1].size())

    batch = []
    for idx, exp in enumerate(exp_source):
        batch.append(exp)
        if len(batch) < BATCH_SIZE:
            continue

        break

    pass
