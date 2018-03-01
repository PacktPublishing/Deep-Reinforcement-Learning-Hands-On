#!/usr/bin/env python3
import ptan
import gym
import numpy as np
import argparse

from lib import common, i2a

import torch
import torch.nn.functional as F
import torchvision

ROLLOUTS_STEPS = 20


def rollouts(n_actions, obs, net_policy, net_em, cuda=False, save=False):
    act_selector = ptan.actions.ProbabilityActionSelector()
    obs_v = ptan.agent.default_states_preprocessor([obs]*n_actions, cuda=cuda)
    obs_v = obs_v.float() / 255
    actions = np.arange(0, n_actions, dtype=np.int64)

    for step_idx in range(ROLLOUTS_STEPS):
        actions_t = torch.from_numpy(actions)
        if cuda:
            actions_t = actions_t.cuda()
        obs_next_v, reward_v = net_em(obs_v, actions_t)
        if save:
            in_images = [obs_v.data[i, 0].t().numpy() for i in range(n_actions)]
            in_images = torch.from_numpy(np.array(in_images, dtype=np.float32)).unsqueeze(1)
            torchvision.utils.save_image(in_images, "%03d-in.png" % step_idx)
            out_images = [obs_next_v.data[i, 0].t().numpy() for i in range(n_actions)]
            out_images = torch.from_numpy(np.array(out_images, dtype=np.float32)).unsqueeze(1)
            torchvision.utils.save_image(out_images, "%03d-out.png" % step_idx)
        obs_v = obs_next_v
        # select actions
        logits_v, _ = net_policy(obs_v)
        probs_v = F.softmax(logits_v)
        probs = probs_v.data.cpu().numpy()
        actions = act_selector(probs)

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    parser.add_argument("--em", required=True, help="Environment model file name")
    parser.add_argument("--policy", required=True, help="Initial policy network")
    args = parser.parse_args()

    make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("BreakoutNoFrameskip-v4"))
    env = make_env()

    net_policy = common.AtariA2C(env.observation_space.shape, env.action_space.n)
    net_policy.load_state_dict(torch.load(args.policy, map_location=lambda storage, loc: storage))

    net_em = i2a.EnvironmentModel(env.observation_space.shape, env.action_space.n)
    net_em.load_state_dict(torch.load(args.em, map_location=lambda storage, loc: storage))

    if args.cuda:
        net_policy.cuda()
        net_em.cuda()

    obs = env.reset()
    rollouts(env.action_space.n, obs, net_policy, net_em, args.cuda, save=True)

    pass
