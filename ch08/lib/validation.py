import numpy as np

import torch
from torch.autograd import Variable

from lib import environ


def validation_run(env, net, episodes=100, cuda=False, epsilon=0.02, comission=0.1):
    stats = {
        'total_reward': [],
        'real_profit_perc': [],
        'order_profits': [],
        'order_steps': [],
        'episode_steps': [],
    }

    for episode in range(episodes):
        obs = env.reset()
        start_price = env._state._cur_close()

        total_reward = 0.0
        real_profit = 0.0
        position = None
        position_steps = None
        episode_steps = 0

        while True:
            obs_v = Variable(torch.from_numpy(np.expand_dims(obs, 0)))
            if cuda:
                obs_v = obs_v.cuda()
            out_v = net(obs_v)

            action_idx = out_v.max(dim=1)[1].data.cpu().numpy()[0]
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            action = environ.Actions(action_idx)

            close_price = env._state._cur_close()

            if action == environ.Actions.Buy and position is None:
                position = close_price
                position_steps = 0
                real_profit -= close_price * comission / 100
            elif action == environ.Actions.Close and position is not None:
                real_profit -= close_price * comission / 100
                real_profit += close_price - position
                stats['order_profits'].append(close_price - position - (close_price + position) * comission / 100)
                stats['order_steps'].append(position_steps)
                position = None
                position_steps = None

            obs, reward, done, _ = env.step(action_idx)
            total_reward += reward
            episode_steps += 1
            if position_steps is not None:
                position_steps += 1
            if done:
                if position is not None:
                    real_profit -= close_price * comission / 100
                    real_profit += close_price - position
                    stats['order_profits'].append(close_price - position - (close_price + position) * comission / 100)
                    stats['order_steps'].append(position_steps)
                break

        real_profit_perc = 100.0 * real_profit / start_price
        stats['total_reward'].append(total_reward)
        stats['real_profit_perc'].append(real_profit_perc)
        stats['episode_steps'].append(episode_steps)

    return { key: np.mean(vals) for key, vals in stats.items() }
