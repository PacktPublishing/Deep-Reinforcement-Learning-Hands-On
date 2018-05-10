#!/usr/bin/env python3
import ptan
import gym
import argparse
import numpy as np

from lib import common

import torch
import torch.nn.functional as F


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file name")
    parser.add_argument("-w", "--write", required=True, help="Monitor directory name")
    parser.add_argument("--cuda", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("BreakoutNoFrameskip-v4"),
                                                     stack_frames=common.FRAMES_COUNT,
                                                     episodic_life=False, reward_clipping=False)
    env = make_env()
    env = gym.wrappers.Monitor(env, args.write)
    net = common.AtariA2C(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    if args.cuda:
        net.cuda()

    act_selector = ptan.actions.ProbabilityActionSelector()

    obs = env.reset()
    total_reward = 0.0
    total_steps = 0

    while True:
        obs_v = ptan.agent.default_states_preprocessor([obs]).to(device)
        logits_v, values_v = net(obs_v)
        probs_v = F.softmax(logits_v)
        probs = probs_v.data.cpu().numpy()
        actions = act_selector(probs)
        obs, r, done, _ = env.step(actions[0])
        total_reward += r
        total_steps += 1
        if done:
            break

    print("Done in %d steps, reward %.2f" % (total_steps, total_reward))
