#!/usr/bin/env python3
import gym
import ptan
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.001
BATCH_SIZE = 32

REWARD_STEPS = 4
BASELINE_STEPS = 100000


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))


class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()

    envs = [make_env() for _ in range(50)]
    writer = SummaryWriter(comment="-pong-a2c")

    net = AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n)
    if args.cuda:
        net.cuda()
    print(net)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, cuda=args.cuda)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    total_rewards = []
    step_idx = 0
    done_episodes = 0
    train_step_idx = 0

    batch_states, batch_actions, batch_rewards, batch_last_states, batch_dones = [], [], [], [], []
    m_values, m_batch_rewards, m_loss_entropy, m_loss_policy, m_loss_total, m_loss_value = [], [], [], [], [], []
    m_adv = []
    m_grad_max, m_grad_mean = [], []

    with common.RewardTracker(writer, stop_reward=18) as tracker:
        for step_idx, exp in enumerate(exp_source):
            batch_states.append(np.array(exp.state, copy=False))
            batch_actions.append(int(exp.action))
            batch_rewards.append(exp.reward)
            batch_dones.append(exp.last_state is None)
            if exp.last_state is not None:
                batch_last_states.append(np.array(exp.last_state, copy=False))
            # handle new rewards
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if tracker.reward(new_rewards[0], step_idx):
                    break

            if len(batch_states) < BATCH_SIZE:
                continue

            train_step_idx += 1
            states_v = Variable(torch.from_numpy(np.array(batch_states, copy=False)))
            if not all(batch_dones):
                last_states_v = Variable(torch.from_numpy(np.array(batch_last_states, copy=False)))
            else:
                last_states_v = None
            batch_actions_t = torch.LongTensor(batch_actions)
            if args.cuda:
                states_v = states_v.cuda()
                if last_states_v is not None:
                    last_states_v = last_states_v.cuda()
                batch_actions_t = batch_actions_t.cuda()

            batch_rewards_np = np.array(batch_rewards, dtype=np.float32)
            if last_states_v is not None:
                last_values = net(last_states_v)[1].data.cpu().numpy()
                batch_dones_np = np.array(batch_dones)
                batch_rewards_np[~batch_dones_np] += (GAMMA ** REWARD_STEPS) * last_values[:, 0]

            batch_rewards_v = Variable(torch.from_numpy(batch_rewards_np))
            if args.cuda:
                batch_rewards_v = batch_rewards_v.cuda()

            optimizer.zero_grad()
            logits_v, value_v = net(states_v)

            loss_value_v = F.mse_loss(value_v, batch_rewards_v)

            log_prob_v = F.log_softmax(logits_v)
            adv_v = batch_rewards_v - value_v.detach()
            log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
            loss_policy_v = -log_prob_actions_v.mean()


            prob_v = F.softmax(logits_v)
            entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()
            loss_v = loss_policy_v + entropy_loss_v + loss_value_v
            loss_v.backward()
            optimizer.step()

            m_adv.append(adv_v.mean().data.cpu().numpy()[0])
            m_values.append(value_v.mean().data.cpu().numpy()[0])
            m_batch_rewards.append(np.mean(batch_rewards))
            m_loss_entropy.append(entropy_loss_v.data.cpu().numpy()[0])
            m_loss_policy.append(loss_policy_v.data.cpu().numpy()[0])
            m_loss_total.append(loss_v.data.cpu().numpy()[0])
            m_loss_value.append(loss_value_v.data.cpu().numpy()[0])

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
                writer.add_scalar("advantage", np.mean(m_adv), step_idx)
                writer.add_scalar("values", np.mean(m_values), step_idx)
                writer.add_scalar("batch_rewards", np.mean(m_batch_rewards), step_idx)
                writer.add_scalar("loss_entropy", np.mean(m_loss_entropy), step_idx)
                writer.add_scalar("loss_policy", np.mean(m_loss_policy), step_idx)
                writer.add_scalar("loss_value", np.mean(m_loss_value), step_idx)
                writer.add_scalar("loss_total", np.mean(m_loss_total), step_idx)
                writer.add_scalar("grad_mean", np.mean(m_grad_mean), step_idx)
                writer.add_scalar("grad_max", np.max(m_grad_max), step_idx)
                m_values, m_batch_rewards, m_loss_entropy, m_loss_total, m_loss_policy = [], [], [], [], []
                m_adv = []
                m_grad_max, m_grad_mean = [], []

            batch_states.clear()
            batch_actions.clear()
            batch_rewards.clear()
            batch_last_states.clear()
            batch_dones.clear()

    writer.close()
