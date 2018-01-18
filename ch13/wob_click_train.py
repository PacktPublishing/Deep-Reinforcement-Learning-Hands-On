#!/usr/bin/env python3
import gym
import universe
import argparse
import numpy as np

from lib import wob_vnc, model_vnc

import torch
from torch.autograd import Variable


ENV_NAME = "wob.mini.BisectAngle-v0"
REMOTE_ADDR = 'vnc://gpu:5900+15900'


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
    args = parser.parse_args()

    env = gym.make(ENV_NAME)
    env = universe.wrappers.experimental.SoftmaxClickMouse(env)
    env = wob_vnc.MiniWoBCropper(env)

    env.configure(remotes=REMOTE_ADDR)
    obs = env.reset()

    net = model_vnc.Model(input_shape=(3, wob_vnc.HEIGHT, wob_vnc.WIDTH))
    print(net)

    obs, reward, done, info = step_env(env, env.action_space.sample())
    obs_v = Variable(torch.from_numpy(np.array(obs, dtype=np.float32)))
    print(obs_v)
    r = net(obs_v)
    print(r.size())
    

    pass
