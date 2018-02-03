#!/usr/bin/env python3
import os
import math
import ptan
import gym
import pybullet_envs
import argparse
from tensorboardX import SummaryWriter

from lib import model, common

import torch
import torch.optim as optim
import torch.nn.functional as F


ENV_ID = "MinitaurBulletEnv-v0"
GAMMA = 0.99
REWARD_STEPS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100000
REPLAY_INITIAL = 1000


def calc_logprob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    save_path = os.path.join("saves", args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_ID)

    net = model.ModelDDPG(env.observation_space.shape[0], env.action_space.shape[0])
    if args.cuda:
        net.cuda()
    print(net)

    writer = SummaryWriter(comment="-ddpg_" + args.name)
    agent = model.AgentDDPG(net, cuda=args.cuda)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    with ptan.common.utils.RewardTracker(writer) as tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            rewards = exp_source.pop_total_rewards()
            if rewards:
                tracker.reward(rewards[0], frame_idx)

            if len(buffer) < REPLAY_INITIAL:
                continue

            batch = buffer.sample(BATCH_SIZE)
            print(len(batch))
            break

    # batch = []
    # best_reward = None
    # with ptan.common.utils.RewardTracker(writer) as tracker:
    #     with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
    #         for step_idx, exp in enumerate(exp_source):
    #             rewards = exp_source.pop_total_rewards()
    #             if rewards:
    #                 mean_reward = tracker.reward(rewards[0], step_idx)
    #                 if mean_reward is not None and (best_reward is None or best_reward < mean_reward):
    #                     if best_reward is not None:
    #                         print("Best reward updated: %.3f -> %.3f" % (best_reward, mean_reward))
    #                         name = "best_%+.3f_%d.dat" % (mean_reward, step_idx)
    #                         fname = os.path.join(save_path, name)
    #                         torch.save(net.state_dict(), fname)
    #                     best_reward = mean_reward
    #
    #             batch.append(exp)
    #             if len(batch) < BATCH_SIZE:
    #                 continue
    #
    #             states_v, actions_v, vals_ref_v = \
    #                 common.unpack_batch(batch, net, last_val_gamma=GAMMA ** REWARD_STEPS, cuda=args.cuda)
    #             batch.clear()
    #
    #             optimizer.zero_grad()
    #             mu_v, var_v, value_v = net(states_v)
    #
    #             loss_value_v = F.mse_loss(value_v, vals_ref_v)
    #
    #             adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
    #             log_prob_v = adv_v * calc_logprob(mu_v, var_v, actions_v)
    #             loss_policy_v = -log_prob_v.mean()
    #             entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*var_v) + 1)/2).mean()
    #
    #             loss_v = loss_policy_v + entropy_loss_v + loss_value_v
    #             loss_v.backward()
    #             optimizer.step()
    #
    #             tb_tracker.track("advantage", adv_v, step_idx)
    #             tb_tracker.track("values", value_v, step_idx)
    #             tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
    #             tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
    #             tb_tracker.track("loss_policy", loss_policy_v, step_idx)
    #             tb_tracker.track("loss_value", loss_value_v, step_idx)
    #             tb_tracker.track("loss_total", loss_v, step_idx)
