import random


class Environment:
    def __init__(self):
        self.steps_left = 10

    def get_observation(self):
        return [0.0, 0.0, 0.0]

    def get_actions(self):
        return [0, 1]

    def is_done(self):
        return self.steps_left == 0

    def action(self, action):
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        return random.random()


class Agent:
    def __init__(self):
        self.total_reward = 0.0

    def step(self, env):
        current_obs = env.get_observation()
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward


if __name__ == "__main__":
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)

    print("Total reward got: %.4f" % agent.total_reward)
