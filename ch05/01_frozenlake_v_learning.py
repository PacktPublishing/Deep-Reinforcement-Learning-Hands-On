#!/usr/bin/env python3
import gym
import collections
from tensorboardX import SummaryWriter

GAMMA = 0.9
TEST_EPISODES = 100


class Agent:
    def __init__(self, env):
        self.env = env
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def select_action(self, state, values):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            target_counts = self.transits[(state, action)]
            total = sum(target_counts.values())
            action_value = 0.0
            for tgt_state, count in target_counts.items():
                reward = self.rewards[(state, action, tgt_state)]
                action_value += (count / total) * (reward + GAMMA * values[tgt_state])
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        if best_action is not None:
            return best_action
        return self.env.action_space.sample()

    def play_episode(self, values):
        total_reward = 0.0
        state = self.env.reset()
        while True:
            action = self.select_action(state, values)
            state, reward, is_done, _ = self.env.step(action)
            total_reward += reward
            if is_done:
                break
        return total_reward


def value_iteration(values, agent):
    assert isinstance(agent, Agent)
    for state in range(agent.env.observation_space.n):
        state_values = []
        for action in range(agent.env.action_space.n):
            action_value = 0.0
            target_counts = agent.transits[(state, action)]
            total = sum(target_counts.values())
            for tgt_state, count in target_counts.items():
                reward = agent.rewards[(state, action, tgt_state)]
                action_value += (count / total) * (reward + GAMMA * values[tgt_state])
            state_values.append(action_value)
        values[state] = max(state_values)


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    agent = Agent(env)
    writer = SummaryWriter(comment="-v-learning")
    values = collections.defaultdict(float)

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        value_iteration(values, agent)

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(values)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.78:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
