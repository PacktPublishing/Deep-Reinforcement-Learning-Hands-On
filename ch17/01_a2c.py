#!/usr/bin/env python3
import gym
import ptan
import argparse
from tensorboardX import SummaryWriter

import torch.optim as optim

from lib import common


LEARNING_RATE = 0.001
NUM_ENVS = 50

REWARD_BOUND = 400


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("BreakoutNoFrameskip-v4"))
    envs = [make_env() for _ in range(NUM_ENVS)]
    writer = SummaryWriter(comment="-breakout-a2c_" + args.name)

    net = common.AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n)
    if args.cuda:
        net.cuda()
    print(net)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, cuda=args.cuda)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=common.GAMMA, steps_count=common.REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    batch = []

    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                # handle new rewards
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    mean_reward = tracker.reward(new_rewards[0], step_idx)
                    if mean_reward is not None and mean_reward > REWARD_BOUND:
                        print("Solved in %d steps" % step_idx)
                        break

                if len(batch) < common.BATCH_SIZE:
                    continue
                states_v, actions_t, vals_ref_v = common.unpack_batch(batch, net, cuda=args.cuda)
                common.train_on_batch(step_idx, net, optimizer, tb_tracker,
                                      states_v, actions_t, vals_ref_v,
                                      cuda=args.cuda)
                batch.clear()


