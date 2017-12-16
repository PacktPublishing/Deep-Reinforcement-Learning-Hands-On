#!/usr/bin/env python3
import gym
import ptan
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ADAM_EPS = 1e-3
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1


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


def unpack_batch(batch, net, cuda=False):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))
    states_v = Variable(torch.from_numpy(np.array(states, copy=False)))
    actions_t = torch.LongTensor(actions)
    if cuda:
        states_v = states_v.cuda()
        actions_t = actions_t.cuda()

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = Variable(torch.from_numpy(np.array(last_states, copy=False)), volatile=True)
        if cuda:
            last_states_v = last_states_v.cuda()
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_vals_np

    ref_vals_v = Variable(torch.from_numpy(rewards_np))
    if cuda:
        ref_vals_v = ref_vals_v.cuda()

    return states_v, actions_t, ref_vals_v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    envs = [make_env() for _ in range(NUM_ENVS)]
    writer = SummaryWriter(comment="-pong-a2c_" + args.name)

    net = AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n)
    if args.cuda:
        net.cuda()
    print(net)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, cuda=args.cuda)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)

    total_rewards = []
    step_idx = 0
    done_episodes = 0
    train_step_idx = 0

    batch = []
    m_values, m_batch_rewards, m_loss_entropy, m_loss_policy, m_loss_total, m_loss_value = [], [], [], [], [], []
    m_adv = []

    with common.RewardTracker(writer, stop_reward=18) as tracker:
        for step_idx, exp in enumerate(exp_source):
            batch.append(exp)

            # handle new rewards
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if tracker.reward(new_rewards[0], step_idx):
                    break

            if len(batch) < BATCH_SIZE:
                continue

            train_step_idx += 1
            states_v, actions_t, vals_ref_v = unpack_batch(batch, net, cuda=args.cuda)

            optimizer.zero_grad()
            logits_v, value_v = net(states_v)

            loss_value_v = F.mse_loss(value_v, vals_ref_v)

            log_prob_v = F.log_softmax(logits_v)
            adv_v = vals_ref_v - value_v.detach()
            log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
            loss_policy_v = -log_prob_actions_v.mean()

            prob_v = F.softmax(logits_v)
            entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()
#            loss_v = loss_policy_v + entropy_loss_v + loss_value_v
            loss_v = loss_value_v
            loss_v.backward()
            nn_utils.clip_grad_norm(net.parameters(), CLIP_GRAD)
            optimizer.step()

            m_adv.append(adv_v.mean().data.cpu().numpy()[0])
            m_values.append(value_v.mean().data.cpu().numpy()[0])
            m_batch_rewards.append(vals_ref_v.mean().data.cpu().numpy()[0])
            m_loss_entropy.append(entropy_loss_v.data.cpu().numpy()[0])
            m_loss_policy.append(loss_policy_v.data.cpu().numpy()[0])
            m_loss_total.append(loss_v.data.cpu().numpy()[0])
            m_loss_value.append(loss_value_v.data.cpu().numpy()[0])

            grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                    for p in net.parameters()
                                    if p.grad is not None])

            if train_step_idx % 10 == 0:
                writer.add_scalar("advantage", np.mean(m_adv), step_idx)
                writer.add_scalar("values", np.mean(m_values), step_idx)
                writer.add_scalar("batch_rewards", np.mean(m_batch_rewards), step_idx)
                writer.add_scalar("loss_entropy", np.mean(m_loss_entropy), step_idx)
                writer.add_scalar("loss_policy", np.mean(m_loss_policy), step_idx)
                writer.add_scalar("loss_value", np.mean(m_loss_value), step_idx)
                writer.add_scalar("loss_total", np.mean(m_loss_total), step_idx)

                writer.add_scalar("grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx)
                writer.add_scalar("grad_max", np.max(np.abs(grads)), step_idx)
                writer.add_scalar("grad_var", np.var(grads), step_idx)

                m_values, m_batch_rewards, m_loss_entropy, m_loss_total, m_loss_policy = [], [], [], [], []
                m_adv, m_loss_value = [], []

            batch.clear()

    writer.close()
