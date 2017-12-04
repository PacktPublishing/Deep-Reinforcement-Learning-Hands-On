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
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 32

REWARD_STEPS = 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()

    env = gym.make("PongNoFrameskip-v4")
    env = ptan.common.wrappers.wrap_dqn(env)
    writer = SummaryWriter(comment="-pong-pg")

    net = common.AtariPGN(env.observation_space.shape, env.action_space.n)
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

    with common.RewardTracker(writer, stop_reward=18) as tracker:
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
            if new_rewards and tracker.reward(new_rewards[0], step_idx):
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

            prob_v = F.softmax(logits_v)
            entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum()
            writer.add_scalar("loss_entropy", entropy_loss_v.data.cpu().numpy()[0], step_idx)
            loss_v += entropy_loss_v

            loss_v.backward()
            optimizer.step()

            batch_states.clear()
            batch_actions.clear()
            batch_scales.clear()

    writer.close()
