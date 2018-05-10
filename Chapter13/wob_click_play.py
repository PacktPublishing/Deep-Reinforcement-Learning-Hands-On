#!/usr/bin/env python3
import argparse
import gym
import universe
import numpy as np

import torch
import torch.nn.functional as F

from lib import wob_vnc, model_vnc


ENV_NAME = "wob.mini.ClickDialog-v0"
REMOTE_ADDR = 'vnc://localhost:5900+15900'


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
    parser.add_argument("--save", help="Enables screenshots and gives an images prefix")
    parser.add_argument("--count", type=int, default=1, help="Count of episodes to play, default=1")
    parser.add_argument("--env", default=ENV_NAME, help="Environment name to solve, default=" + ENV_NAME)
    parser.add_argument("--verbose", default=False, action='store_true', help="Display every step")
    args = parser.parse_args()

    env_name = args.env
    if not env_name.startswith('wob.mini.'):
        env_name = "wob.mini." + env_name

    env = gym.make(env_name)
    env = universe.wrappers.experimental.SoftmaxClickMouse(env)
    if args.save is not None:
        env = wob_vnc.MiniWoBPeeker(env, args.save)
    env = wob_vnc.MiniWoBCropper(env)
    wob_vnc.configure(env, REMOTE_ADDR, fps=5)

    net = model_vnc.Model(input_shape=wob_vnc.WOB_SHAPE, n_actions=env.action_space.n)
    if args.model:
        net.load_state_dict(torch.load(args.model))

    env.reset()
    steps_count = 0
    reward_sum = 0

    for round_idx in range(args.count):
        action = env.action_space.sample()
        step_idx = 0
        while True:
            obs, reward, done, info, idle_count = step_env(env, action)
            if args.verbose:
                print(step_idx, reward, done, idle_count, info)
            obs_v = torch.tensor(obs)
            logits_v = net(obs_v)[0]
            policy = F.softmax(logits_v, dim=1).data.numpy()[0]
            action = np.random.choice(len(policy), p=policy)
            step_idx += 1
            reward_sum += reward
            steps_count += 1
            if done or reward != 0:
                print("Round %d done" % round_idx)
                break
    print("Done %d rounds, mean steps %.2f, mean reward %.3f" % (
        args.count, steps_count / args.count, reward_sum / args.count
    ))

    pass
