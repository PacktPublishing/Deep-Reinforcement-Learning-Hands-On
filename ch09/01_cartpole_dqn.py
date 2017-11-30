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
LEARNING_RATE = 0.005
BATCH_SIZE = 8

EPSILON_START = 1.0
EPSILON_STOP = 0.1
EPSILON_STEPS = 50000

REPLAY_BUFFER = 10000

TARGET_STEPS = 2000


class DQN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


def calc_target(net, local_reward, next_state):
    if next_state is None:
        return local_reward
    state_v = Variable(torch.from_numpy(np.array([next_state], dtype=np.float32)))
    next_q_v = net(state_v)
    best_q = next_q_v.max(dim=1)[0].data.numpy()[0]
    return local_reward + GAMMA * best_q


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="cartpole-dqn")

    net = DQN(env.observation_space.shape[0], env.action_space.n)
    tgt_net = ptan.agent.TargetNet(net)
    print(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    replay_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_BUFFER)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    mse_loss = nn.MSELoss()

    batch_states, batch_actions, batch_targets = [], [], []
    total_rewards = []

    for step_idx, exp in enumerate(exp_source):
        selector.epsilon = max(EPSILON_STOP, EPSILON_START - step_idx / EPSILON_STEPS)
        batch_states.append(exp.state)
        batch_actions.append(exp.action)
        batch_targets.append(calc_target(tgt_net.target_model, exp.reward, exp.last_state))
        if len(batch_states) == BATCH_SIZE:
            optimizer.zero_grad()
            states_v = Variable(torch.from_numpy(np.array(batch_states, dtype=np.float32)))
            net_q_v = net(states_v)
            target_q = net_q_v.data.numpy().copy()
            target_q[range(BATCH_SIZE), batch_actions] = batch_targets
            target_q_v = Variable(torch.from_numpy(target_q))
            loss_v = mse_loss(net_q_v, target_q_v)
            loss_v.backward()
            optimizer.step()

            # clear batch
            batch_states.clear()
            batch_actions.clear()
            batch_targets.clear()

        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f,\tmean_100: %6.2f,\tepsilon: %.2f" % (step_idx, reward, mean_rewards, selector.epsilon))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("epsilon", selector.epsilon, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps!" % step_idx)
                break

        if step_idx % TARGET_STEPS == 0:
            tgt_net.sync()
    writer.close()
    pass
