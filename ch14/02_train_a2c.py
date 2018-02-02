#!/usr/bin/env python3
import ptan
import gym
import math
import pybullet_envs
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
ENTROPY_BETA = 1e-4


def calc_logprob(mu_v, var_v, actions_v):
    res = - ((mu_v - actions_v) ** 2) / (2*var_v)
    res -= torch.log(torch.sqrt(2 * math.pi * var_v))
    return res


if __name__ == "__main__":
    env = gym.make(ENV_ID)

    net = model.Model(env.observation_space.shape[0], env.action_space.shape[0])
    print(net)

    writer = SummaryWriter(comment="a2c")
    agent = model.ContAgent(net)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    batch = []
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                rewards = exp_source.pop_total_rewards()
                if rewards:
                    tracker.reward(rewards[0], step_idx)

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = \
                    common.unpack_batch(batch, net, last_val_gamma=GAMMA ** REWARD_STEPS)
                batch.clear()

                optimizer.zero_grad()
                mu_v, var_v, value_v = net(states_v)

                loss_value_v = F.mse_loss(value_v, vals_ref_v)

                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                log_prob_v = adv_v * calc_logprob(mu_v, var_v, actions_v)
                loss_policy_v = -log_prob_v.mean()
                entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*var_v) + 1)/2).mean()

                loss_v = loss_policy_v + entropy_loss_v + loss_value_v
                loss_v.backward()
                optimizer.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
    pass
