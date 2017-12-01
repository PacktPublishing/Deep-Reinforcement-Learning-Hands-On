#!/usr/bin/env python3
import gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 32

REPLAY_BUFFER = 100
MIN_EPISODES_TO_TRAIN = 10

EPSILON_START = 1.0
EPSILON_STOP = 0.02
EPSILON_STEPS = 50000


class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-pg")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON_START,
                                                        selector=ptan.actions.ProbabilityActionSelector())
    agent = ptan.agent.PolicyAgent(net, action_selector=selector, preprocessor=ptan.agent.float32_preprocessor,
                                   apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=2)
    replay_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_BUFFER)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_idx = 0
    done_episodes = 0
    total_step_reward = 0.0

    for step_idx, exp in enumerate(exp_source):
        selector.epsilon = max(EPSILON_STOP, EPSILON_START - step_idx / EPSILON_STEPS)
        total_step_reward += exp.reward
        replay_buffer._add(exp)

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, epsilon: %.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, selector.epsilon, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("epsilon", selector.epsilon, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        if len(total_rewards) < MIN_EPISODES_TO_TRAIN:
            continue

        batch = replay_buffer.sample(BATCH_SIZE)
        batch_states = [exp.state for exp in batch]
        batch_actions_t = torch.LongTensor([int(exp.action) for exp in batch])
        mean_step_reward = total_step_reward / (step_idx + 1)
        batch_scale_t = torch.FloatTensor([exp.reward - mean_step_reward for exp in batch])

        optimizer.zero_grad()
        states_v = Variable(torch.from_numpy(np.array(batch_states, dtype=np.float32)))
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v)
        log_prob_v = Variable(batch_scale_t) * log_prob_v[range(BATCH_SIZE), batch_actions_t]
        loss_v = log_prob_v.mean()
        loss_v.backward()
        optimizer.step()
    writer.close()
