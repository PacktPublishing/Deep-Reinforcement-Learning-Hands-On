#!/usr/bin/env python3
import argparse
import gym
import universe
import numpy as np

import torch
import torch.nn.functional as F

from lib import wob_vnc, model_vnc


ENV_NAME = "wob.mini.ClickTab-v0"
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
    parser.add_argument("--count", type=int, default=1, help="Count of runs to play, default=1")
    parser.add_argument("--env", default=ENV_NAME, help="Environment name to solve, default=" + ENV_NAME)
    args = parser.parse_args()

    env_name = args.env
    if not env_name.startswith('wob.mini.'):
        env_name = "wob.mini." + env_name

    env = gym.make(env_name)
    env = universe.wrappers.experimental.SoftmaxClickMouse(env)
    env = wob_vnc.MiniWoBCropper(env, keep_text=True)
    wob_vnc.configure(env, REMOTE_ADDR)

    net = model_vnc.ModelMultimodal(input_shape=wob_vnc.WOB_SHAPE, n_actions=env.action_space.n)
    if args.model:
        net.load_state_dict(torch.load(args.model))
        preprocessor = model_vnc.MultimodalPreprocessor.load(args.model[:-4] + ".pre")
    else:
        preprocessor = model_vnc.MultimodalPreprocessor()
    env.reset()

    for round_idx in range(args.count):
        action = env.action_space.sample()
        step_idx = 0
        while True:
            obs, reward, done, info, idle_count = step_env(env, action)
            print(step_idx, reward, done, idle_count)
            img_name = "%s_r%02d_s%04d_%.3f_i%02d_d%d.png" % (
                args.name, round_idx, step_idx, reward, idle_count, int(done))
            obs_v = preprocessor([obs])
            logits_v = net(obs_v)[0]
            policy = F.softmax(logits_v, dim=1).data.numpy()[0]
            action = np.random.choice(len(policy), p=policy)
            wob_vnc.save_obs(obs[0], img_name, action=action)
            step_idx += 1
            if done or reward != 0:
                print("Round %d done" % round_idx)
                break
    pass
