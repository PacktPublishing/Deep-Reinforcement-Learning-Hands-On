#!/usr/bin/env python3
import argparse
import gym
import universe
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image

from lib import wob_vnc, model_vnc


ENV_NAME = "wob.mini.ClickDialog-v0"
REMOTE_ADDR = 'vnc://gpu:5910+15910'

# docker run -d -p 5910:5900 -p 15910:15900 --privileged --ipc host --cap-add SYS_ADMIN 92756d1f08ac


def step_env(env, action):
    idle_count = 0
    while True:
        obs, reward, is_done, info = env.step([action])
        if obs[0] is None:
            idle_count += 1
            continue
        break
    return obs[0], reward[0], is_done[0], info, idle_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model file to load")
    parser.add_argument("-n", "--name", required=True, help="Prefix to save screenshots")
    args = parser.parse_args()

    env = gym.make(ENV_NAME)
    env = universe.wrappers.experimental.SoftmaxClickMouse(env)
    env = wob_vnc.MiniWoBCropper(env)
    env.configure(remotes=REMOTE_ADDR)

    net = model_vnc.Model(input_shape=wob_vnc.WOB_SHAPE, n_actions=env.action_space.n)
    if args.model:
        net.load_state_dict(torch.load(args.model))

    env.reset()
    action = env.action_space.sample()
    step_idx = 0

    while True:
        obs, reward, done, info, idle_count = step_env(env, action)
        print(step_idx, reward, done)
        if done or reward != 0:
            break
        img = Image.fromarray(np.transpose(obs, (1, 2, 0)))
        img.save("%s_%04d_%.3f.png" % (args.name, step_idx, reward))
        obs_v = Variable(torch.from_numpy(np.array([obs])))
        logits_v = net(obs_v)[0]
        policy = F.softmax(logits_v).data.numpy()[0]
        action = np.random.choice(len(policy), p=policy)
        step_idx += 1
    pass
