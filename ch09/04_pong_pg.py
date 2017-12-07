#!/usr/bin/env python3
import gym
import ptan
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim
from torch.autograd import Variable

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128

REWARD_STEPS = 100
BASELINE_STEPS = 100000
GRAD_L2_CLIP = 0.1

ENV_COUNT = 32


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))


class MeanRingBuf:
    def __init__(self, capacity):
        self.capacity = capacity
        self.full = False
        self.pos = 0
        self._buf = np.zeros((capacity, ), dtype=np.float32)

    def add(self, val):
        self._buf[self.pos] = val
        self.pos = (self.pos + 1) % self.capacity
        self.full |= self.pos == 0

    def mean(self):
        if self.full:
            return self._buf.mean()
        else:
            return self._buf[:self.pos].mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", '--name', required=True, help="Name of the run")
    args = parser.parse_args()

    envs = [make_env() for _ in range(ENV_COUNT)]
    writer = SummaryWriter(comment="-pong-pg-" + args.name)

    net = common.AtariPGN(envs[0].observation_space.shape, envs[0].action_space.n)
    if args.cuda:
        net.cuda()
    print(net)

    agent = ptan.agent.PolicyAgent(net, apply_softmax=True, cuda=args.cuda)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    total_rewards = []
    step_idx = 0
    done_episodes = 0
    train_step_idx = 0
    baseline_buf = MeanRingBuf(BASELINE_STEPS)

    batch_states, batch_actions, batch_scales = [], [], []
    m_baseline, m_batch_scales, m_loss_entropy, m_loss_policy, m_loss_total = [], [], [], [], []
    m_grad_max, m_grad_mean = [], []
    sum_reward = 0.0

    with common.RewardTracker(writer, stop_reward=18) as tracker:
        for step_idx, exp in enumerate(exp_source):
            baseline_buf.add(exp.reward)
            baseline = baseline_buf.mean()
            batch_states.append(np.array(exp.state, copy=False))
            batch_actions.append(int(exp.action))
            batch_scales.append(exp.reward - baseline)

            # handle new rewards
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if tracker.reward(new_rewards[0], step_idx):
                    break

            if len(batch_states) < BATCH_SIZE:
                continue

            train_step_idx += 1
            states_v = Variable(torch.from_numpy(np.array(batch_states, copy=False)))
            batch_actions_t = torch.LongTensor(batch_actions)

            scale_std = np.std(batch_scales)
            batch_scale_v = Variable(torch.FloatTensor(batch_scales))
            if args.cuda:
                states_v = states_v.cuda()
                batch_actions_t = batch_actions_t.cuda()
                batch_scale_v = batch_scale_v.cuda()

            optimizer.zero_grad()
            logits_v = net(states_v)
            log_prob_v = F.log_softmax(logits_v)
            log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
            loss_policy_v = -log_prob_actions_v.mean()

            prob_v = F.softmax(logits_v)
            entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
            entropy_loss_v = -ENTROPY_BETA * entropy_v
            loss_v = loss_policy_v + entropy_loss_v
            loss_v.backward()
            nn_utils.clip_grad_norm(net.parameters(), GRAD_L2_CLIP)
            optimizer.step()

            # calc KL-div
            new_logits_v = net(states_v)
            new_prob_v = F.softmax(new_logits_v)
            kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
            writer.add_scalar("kl", kl_div_v.data.cpu().numpy()[0], step_idx)

            grad_max = 0.0
            grad_means = 0.0
            grad_count = 0
            for p in net.parameters():
                grad_max = max(grad_max, p.grad.abs().max().data.cpu().numpy()[0])
                grad_means += (p.grad ** 2).mean().sqrt().data.cpu().numpy()[0]
                grad_count += 1

            writer.add_scalar("baseline", baseline, step_idx)
            writer.add_scalar("entropy", entropy_v.data.cpu().numpy()[0], step_idx)
            writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
            writer.add_scalar("batch_scales_std", scale_std, step_idx)
            writer.add_scalar("loss_entropy", entropy_loss_v.data.cpu().numpy()[0], step_idx)
            writer.add_scalar("loss_policy", loss_policy_v.data.cpu().numpy()[0], step_idx)
            writer.add_scalar("loss_total", loss_v.data.cpu().numpy()[0], step_idx)
            writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
            writer.add_scalar("grad_max", grad_max, step_idx)

            batch_states.clear()
            batch_actions.clear()
            batch_scales.clear()

    writer.close()
