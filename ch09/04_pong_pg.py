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

GAMMA = 0.99
LEARNING_RATE = 0.0001
BATCH_SIZE = 32

REWARD_STEPS = 10


class PGN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PGN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()

    env = gym.make("PongNoFrameskip-v4")
    env = ptan.common.wrappers.wrap_dqn(env)
    writer = SummaryWriter(comment="-pong-pg")

    net = PGN(env.observation_space.shape, env.action_space.n)
    if args.cuda:
        net.cuda()
    print(net)

    agent = ptan.agent.PolicyAgent(net, apply_softmax=True, cuda=args.cuda)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_rewards = []
    step_idx = 0
    done_episodes = 0

    batch_states, batch_actions, batch_scales = [], [], []

    for step_idx, exp in enumerate(exp_source):
        step_rewards.append(exp.reward)
        step_rewards = step_rewards[-1000:]

        baseline = np.mean(step_rewards)
        writer.add_scalar("baseline", baseline, step_idx)
        batch_states.append(np.array(exp.state, copy=False))
        batch_actions.append(int(exp.action))
        batch_scales.append(exp.reward - baseline)

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 18.0:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        if len(batch_states) < BATCH_SIZE:
            continue

        states_v = Variable(torch.from_numpy(np.array(batch_states, copy=False)))
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scale_v = Variable(torch.FloatTensor(batch_scales))
        if args.cuda:
            states_v = states_v.cuda()
            batch_actions_t = batch_actions_t.cuda()
            batch_scale_v = batch_scale_v.cuda()

        optimizer.zero_grad()
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v)
        log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
        loss_v = -log_prob_actions_v.mean()

        loss_v.backward()
        optimizer.step()

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    writer.close()
