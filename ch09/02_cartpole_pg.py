#!/usr/bin/env python3
import gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

GAMMA = 0.99
LEARNING_RATE = 0.01
BATCH_SIZE = 8

REPLAY_BUFFER = 100
MIN_EPISODES_TO_TRAIN = 10


class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax()
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-pg")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    replay_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_BUFFER)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_idx = 0
    done_episodes = 0
    total_step_reward = 0.0

    for step_idx, exp in enumerate(exp_source):
        total_step_reward += exp.reward
        replay_buffer._add(exp)

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
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        if len(total_rewards) < MIN_EPISODES_TO_TRAIN:
            continue

        batch = replay_buffer.sample(BATCH_SIZE)
        batch_states = [exp.state for exp in batch]
        batch_actions_t = torch.LongTensor([int(exp.action) for exp in batch])
        mean_step_reward = total_step_reward / step_idx
        batch_scale_t = torch.FloatTensor([exp.reward - mean_step_reward for exp in batch])

        optimizer.zero_grad()
        states_v = Variable(torch.from_numpy(np.array(batch_states, dtype=np.float32)))
        policy_v = net(states_v)
        prob_actions_v = policy_v[range(BATCH_SIZE), batch_actions_t]
        log_prob_v = torch.log(prob_actions_v)
        log_prob_v = Variable(batch_scale_t) * log_prob_v
        loss_v = log_prob_v.mean()
        loss_v.backward()
        optimizer.step()

        break

    #
    # while True:
    #     step_idx += 1
    #     selector.epsilon = max(EPSILON_STOP, EPSILON_START - step_idx / EPSILON_STEPS)
    #     replay_buffer.populate(1)
    #
    #     if len(replay_buffer) < BATCH_SIZE:
    #         continue
    #
    #     # sample batch
    #     batch = replay_buffer.sample(BATCH_SIZE)
    #     batch_states = [exp.state for exp in batch]
    #     batch_actions = [exp.action for exp in batch]
    #     batch_targets = [calc_target(net, exp.reward, exp.last_state)
    #                      for exp in batch]
    #     # train
    #     optimizer.zero_grad()
    #     states_v = Variable(torch.from_numpy(np.array(batch_states, dtype=np.float32)))
    #     net_q_v = net(states_v)
    #     target_q = net_q_v.data.numpy().copy()
    #     target_q[range(BATCH_SIZE), batch_actions] = batch_targets
    #     target_q_v = Variable(torch.from_numpy(target_q))
    #     loss_v = mse_loss(net_q_v, target_q_v)
    #     loss_v.backward()
    #     optimizer.step()
    #
    #     # handle new rewards
    #     new_rewards = exp_source.pop_total_rewards()
    #     if new_rewards:
    #         done_episodes += 1
    #         reward = new_rewards[0]
    #         total_rewards.append(reward)
    #         mean_rewards = float(np.mean(total_rewards[-100:]))
    #         print("%d: reward: %6.2f, mean_100: %6.2f, epsilon: %.2f, episodes: %d" % (
    #             step_idx, reward, mean_rewards, selector.epsilon, done_episodes))
    #         writer.add_scalar("reward", reward, step_idx)
    #         writer.add_scalar("reward_100", mean_rewards, step_idx)
    #         writer.add_scalar("epsilon", selector.epsilon, step_idx)
    #         writer.add_scalar("episodes", done_episodes, step_idx)
    #         if mean_rewards > 195:
    #             print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
    #             break
#    writer.close()
