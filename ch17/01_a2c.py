#!/usr/bin/env python3
import os
import ptan
import time
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim

from lib import common


LEARNING_RATE = 1e-4
TEST_EVERY_BATCH = 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--seed", type=int, default=common.DEFAULT_SEED, help="Random seed to use, default=%d" % common.DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=None, help="Limit of training steps, default=disabled")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    saves_path = os.path.join("saves", "01_a2c_" + args.name)
    os.makedirs(saves_path, exist_ok=True)

    envs = [common.make_env() for _ in range(common.NUM_ENVS)]
    if args.seed:
        common.set_seed(args.seed, envs, cuda=args.cuda)
        suffix = "-seed=%d" % args.seed
    else:
        suffix = ""

    test_env = common.make_env(test=True)
    writer = SummaryWriter(comment="-01_a2c_" + args.name + suffix)

    net = common.AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)
    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, eps=1e-5)

    step_idx = 0
    total_steps = 0
    best_reward = None
    ts_start = time.time()
    best_test_reward = None
    with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
        for mb_obs, mb_rewards, mb_actions, mb_values, _, done_rewards, done_steps in \
                common.iterate_batches(envs, net, device=device):
            if len(done_rewards) > 0:
                total_steps += sum(done_steps)
                speed = total_steps / (time.time() - ts_start)
                if best_reward is None:
                    best_reward = done_rewards.max()
                elif best_reward < done_rewards.max():
                    best_reward = done_rewards.max()
                tb_tracker.track("total_reward_max", best_reward, step_idx)
                tb_tracker.track("total_reward", done_rewards, step_idx)
                tb_tracker.track("total_steps", done_steps, step_idx)
                print("%d: done %d episodes, mean_reward=%.2f, best_reward=%.2f, speed=%.2f" % (
                    step_idx, len(done_rewards), done_rewards.mean(), best_reward, speed))

            common.train_a2c(net, mb_obs, mb_rewards, mb_actions, mb_values,
                             optimizer, tb_tracker, step_idx, device=device)
            step_idx += 1
            if args.steps is not None and args.steps < step_idx:
                break

            if step_idx % TEST_EVERY_BATCH == 0:
                test_reward, test_steps = common.test_model(test_env, net, device=device)
                writer.add_scalar("test_reward", test_reward, step_idx)
                writer.add_scalar("test_steps", test_steps, step_idx)
                if best_test_reward is None or best_test_reward < test_reward:
                    if best_test_reward is not None:
                        fname = os.path.join(saves_path, "best_%08.3f_%d.dat" % (test_reward, step_idx))
                        torch.save(net.state_dict(), fname)
                    best_test_reward = test_reward
                print("%d: test reward=%.2f, steps=%.2f, best_reward=%.2f" % (
                    step_idx, test_reward, test_steps, best_test_reward))
