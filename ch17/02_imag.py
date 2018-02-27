#!/usr/bin/env python3
import gym
import ptan
import argparse
from tensorboardX import SummaryWriter

import torch.optim as optim
import torch.nn.functional as F

from lib import common, i2a


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
    net_imag_policy = i2a.ImagPolicy(envs[0].observation_space.shape, envs[0].action_space.n)
    if args.cuda:
        net.cuda()
        net_imag_policy.cuda()
    print(net)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, cuda=args.cuda)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=common.GAMMA, steps_count=common.REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    opt_imag_policy = optim.Adam(net_imag_policy.parameters(), lr=LEARNING_RATE)

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
                policy_logits_v = common.train_on_batch(step_idx, net, optimizer, tb_tracker,
                                                        states_v, actions_t, vals_ref_v,
                                                        cuda=args.cuda)

                opt_imag_policy.zero_grad()
                imag_policy_logits_v = net_imag_policy(states_v)
                imag_policy_loss_v = -F.log_softmax(imag_policy_logits_v) * F.softmax(policy_logits_v.detach())
                imag_policy_loss_v = imag_policy_loss_v.sum(dim=1).mean()
                tb_tracker.track("imag_policy_loss", imag_policy_loss_v, step_idx)

                batch.clear()


