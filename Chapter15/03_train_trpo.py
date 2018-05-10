#!/usr/bin/env python3
import os
import math
import ptan
import time
import gym
import roboschool
import argparse
from tensorboardX import SummaryWriter

from lib import model, trpo

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


ENV_ID = "RoboschoolHalfCheetah-v1"
GAMMA = 0.99
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 2049
LEARNING_RATE_CRITIC = 1e-3

TRPO_MAX_KL = 0.01
TRPO_DAMPING = 0.1

TEST_ITERS = 1000


def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def calc_logprob(mu_v, logstd_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2


def calc_adv_ref(trajectory, net_crt, states_v, device="cpu"):
    """
    By trajectory calculate advantage and 1-step ref value
    :param trajectory: list of Experience objects
    :param net_crt: critic network
    :return: tuple with advantage numpy array and reference values
    """
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp,) in zip(reversed(values[:-1]), reversed(values[1:]),
                                     reversed(trajectory[:-1])):
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + GAMMA * next_val - val
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv))).to(device)
    ref_v = torch.FloatTensor(list(reversed(result_ref))).to(device)
    return adv_v, ref_v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment id, default=" + ENV_ID)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "trpo-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(args.env)
    test_env = gym.make(args.env)

    net_act = model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    net_crt = model.ModelCritic(env.observation_space.shape[0]).to(device)
    print(net_act)
    print(net_crt)

    writer = SummaryWriter(comment="-trpo_" + args.name)
    agent = model.AgentA2C(net_act, device=device)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)

    opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)

    trajectory = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        for step_idx, exp in enumerate(exp_source):
            rewards_steps = exp_source.pop_rewards_steps()
            if rewards_steps:
                rewards, steps = zip(*rewards_steps)
                writer.add_scalar("episode_steps", np.mean(steps), step_idx)
                tracker.reward(np.mean(rewards), step_idx)

            if step_idx % TEST_ITERS == 0:
                ts = time.time()
                rewards, steps = test_net(net_act, test_env, device=device)
                print("Test done in %.2f sec, reward %.3f, steps %d" % (
                    time.time() - ts, rewards, steps))
                writer.add_scalar("test_reward", rewards, step_idx)
                writer.add_scalar("test_steps", steps, step_idx)
                if best_reward is None or best_reward < rewards:
                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                        name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                        fname = os.path.join(save_path, name)
                        torch.save(net_act.state_dict(), fname)
                    best_reward = rewards

            trajectory.append(exp)
            if len(trajectory) < TRAJECTORY_SIZE:
                continue

            traj_states = [t[0].state for t in trajectory]
            traj_actions = [t[0].action for t in trajectory]
            traj_states_v = torch.FloatTensor(traj_states).to(device)
            traj_actions_v = torch.FloatTensor(traj_actions).to(device)
            traj_adv_v, traj_ref_v = calc_adv_ref(trajectory, net_crt, traj_states_v, device=device)
            mu_v = net_act(traj_states_v)
            old_logprob_v = calc_logprob(mu_v, net_act.logstd, traj_actions_v)

            # normalize advantages
            traj_adv_v = (traj_adv_v - torch.mean(traj_adv_v)) / torch.std(traj_adv_v)

            # drop last entry from the trajectory, an our adv and ref value calculated without it
            trajectory = trajectory[:-1]
            old_logprob_v = old_logprob_v[:-1].detach()
            traj_states_v = traj_states_v[:-1]
            traj_actions_v = traj_actions_v[:-1]
            sum_loss_value = 0.0
            sum_loss_policy = 0.0
            count_steps = 0

            # critic step
            opt_crt.zero_grad()
            value_v = net_crt(traj_states_v)
            loss_value_v = F.mse_loss(value_v.squeeze(-1), traj_ref_v)
            loss_value_v.backward()
            opt_crt.step()

            # actor step
            def get_loss():
                mu_v = net_act(traj_states_v)
                logprob_v = calc_logprob(mu_v, net_act.logstd, traj_actions_v)
                action_loss_v = -traj_adv_v.unsqueeze(dim=-1) * torch.exp(logprob_v - old_logprob_v)
                return action_loss_v.mean()

            def get_kl():
                mu_v = net_act(traj_states_v)
                logstd_v = net_act.logstd
                mu0_v = mu_v.detach()
                logstd0_v = logstd_v.detach()
                std_v = torch.exp(logstd_v)
                std0_v = std_v.detach()
                kl = logstd_v - logstd0_v + (std0_v ** 2 + ((mu0_v - mu_v) ** 2) / (2.0 * std_v ** 2)) - 0.5
                return kl.sum(1, keepdim=True)

            trpo.trpo_step(net_act, get_loss, get_kl, TRPO_MAX_KL, TRPO_DAMPING, device=device)

            trajectory.clear()
            writer.add_scalar("advantage", traj_adv_v.mean().item(), step_idx)
            writer.add_scalar("values", traj_ref_v.mean().item(), step_idx)
            writer.add_scalar("loss_value", loss_value_v.item(), step_idx)

