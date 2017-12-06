#!/usr/bin/env python3
import gym
import ptan
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.00001
ENTROPY_BETA = 0.0001
BATCH_SIZE = 512

REWARD_STEPS = 500
BASELINE_STEPS = 100000


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
    args = parser.parse_args()

    env = make_env()
    writer = SummaryWriter(comment="-pong-pg")

    net = common.AtariPGN(env.observation_space.shape, env.action_space.n)
    if args.cuda:
        net.cuda()
    print(net)

    agent = ptan.agent.PolicyAgent(net, apply_softmax=True, cuda=args.cuda)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    total_rewards = []
    step_rewards = MeanRingBuf(capacity=BASELINE_STEPS)
    step_idx = 0
    done_episodes = 0
    train_step_idx = 0

    batch_states, batch_actions, batch_scales = [], [], []
    m_baseline, m_batch_scales, m_loss_entropy, m_loss_policy, m_loss_total = [], [], [], [], []
    m_grad_max, m_grad_mean = [], []

    with common.RewardTracker(writer, stop_reward=18) as tracker:
        for step_idx, exp in enumerate(exp_source):
            step_rewards.add(exp.reward)

            baseline = step_rewards.mean()
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
            bs = np.array(batch_scales, dtype=np.float32)
            bs -= bs.mean()
            if abs(bs.std()) > 1e-5:
                bs /= bs.std()

            batch_scale_v = Variable(torch.from_numpy(bs))
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
            entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()
            loss_v = loss_policy_v + entropy_loss_v
            loss_v.backward()
            optimizer.step()

            m_baseline.append(baseline)
            m_batch_scales.append(np.mean(batch_scales))
            m_loss_entropy.append(entropy_loss_v.data.cpu().numpy()[0])
            m_loss_policy.append(loss_policy_v.data.cpu().numpy()[0])
            m_loss_total.append(loss_v.data.cpu().numpy()[0])

            grad_max = 0.0
            grad_means = 0.0
            grad_count = 0
            for p in net.parameters():
                grad_max = max(grad_max, p.grad.abs().max().data.cpu().numpy()[0])
                grad_means += p.grad.mean().data.cpu().numpy()[0]
                grad_count += 1
            m_grad_max.append(grad_max)
            m_grad_mean.append(grad_means / grad_count)

            if train_step_idx % 10 == 0:
                writer.add_scalar("baseline", np.mean(m_baseline), step_idx)
                writer.add_scalar("batch_scales", np.mean(m_batch_scales), step_idx)
                writer.add_scalar("loss_entropy", np.mean(m_loss_entropy), step_idx)
                writer.add_scalar("loss_policy", np.mean(m_loss_policy), step_idx)
                writer.add_scalar("loss_total", np.mean(m_loss_total), step_idx)
                writer.add_scalar("grad_mean", np.mean(m_grad_mean), step_idx)
                writer.add_scalar("grad_max", np.max(m_grad_max), step_idx)
                m_baseline, m_batch_scales, m_loss_entropy, m_loss_total, m_loss_policy = [], [], [], [], []
                m_grad_max, m_grad_mean = [], []

            batch_states.clear()
            batch_actions.clear()
            batch_scales.clear()

    writer.close()
