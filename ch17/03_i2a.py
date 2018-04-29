#!/usr/bin/env python3
import os
import ptan
import time
import gym
import argparse

from lib import common, i2a

import torch.optim as optim
import torch
import torch.nn.functional as F

from tensorboardX import SummaryWriter


ROLLOUTS_STEPS = 3
LEARNING_RATE = 1e-4
POLICY_LR = 1e-4
TEST_EVERY_BATCH = 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    parser.add_argument("--em", required=True, help="Environment model file name")
    parser.add_argument("--seed", type=int, default=common.DEFAULT_SEED, help="Random seed to use, default=%d" % common.DEFAULT_SEED)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    saves_path = os.path.join("saves", "03_i2a_" + args.name)
    os.makedirs(saves_path, exist_ok=True)

    envs = [common.make_env() for _ in range(common.NUM_ENVS)]
    test_env = common.make_env(test=True)

    if args.seed:
        common.set_seed(args.seed, envs, cuda=args.cuda)
        suffix = "-seed=%d" % args.seed
    else:
        suffix = ""
    writer = SummaryWriter(comment="-03_i2a_" + args.name + suffix)

    obs_shape = envs[0].observation_space.shape
    act_n = envs[0].action_space.n

    net_policy = common.AtariA2C(obs_shape, act_n).to(device)

    net_em = i2a.EnvironmentModel(obs_shape, act_n)
    net_em.load_state_dict(torch.load(args.em, map_location=lambda storage, loc: storage))
    net_em = net_em.to(device)

    net_i2a = i2a.I2A(obs_shape, act_n, net_em, net_policy, ROLLOUTS_STEPS).to(device)
    print(net_i2a)

    obs = envs[0].reset()
    obs_v = ptan.agent.default_states_preprocessor([obs]).to(device)
    res = net_i2a(obs_v)

    optimizer = optim.RMSprop(net_i2a.parameters(), lr=LEARNING_RATE, eps=1e-5)
    policy_opt = optim.Adam(net_policy.parameters(), lr=POLICY_LR)

    step_idx = 0
    total_steps = 0
    ts_start = time.time()
    best_reward = None
    best_test_reward = None
    with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
        for mb_obs, mb_rewards, mb_actions, mb_values, mb_probs, done_rewards, done_steps in \
                common.iterate_batches(envs, net_i2a, device):
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
                print("%d: done %d episodes, mean_reward=%.2f, best_reward=%.2f, speed=%.2f f/s" % (
                    step_idx, len(done_rewards), done_rewards.mean(), best_reward, speed))

            obs_v = common.train_a2c(net_i2a, mb_obs, mb_rewards, mb_actions, mb_values,
                                     optimizer, tb_tracker, step_idx, device=device)
            # policy distillation
            probs_v = torch.FloatTensor(mb_probs).to(device)
            policy_opt.zero_grad()
            logits_v, _ = net_policy(obs_v)
            policy_loss_v = -F.log_softmax(logits_v, dim=1) * probs_v.view_as(logits_v)
            policy_loss_v = policy_loss_v.sum(dim=1).mean()
            policy_loss_v.backward()
            policy_opt.step()
            tb_tracker.track("loss_distill", policy_loss_v, step_idx)

            step_idx += 1

            if step_idx % TEST_EVERY_BATCH == 0:
                test_reward, test_steps = common.test_model(test_env, net_i2a, device=device)
                writer.add_scalar("test_reward", test_reward, step_idx)
                writer.add_scalar("test_steps", test_steps, step_idx)
                if best_test_reward is None or best_test_reward < test_reward:
                    if best_test_reward is not None:
                        fname = os.path.join(saves_path, "best_%08.3f_%d.dat" % (test_reward, step_idx))
                        torch.save(net_i2a.state_dict(), fname)
                        torch.save(net_policy.state_dict(), fname + ".policy")
                    else:
                        fname = os.path.join(saves_path, "em.dat")
                        torch.save(net_em.state_dict(), fname)
                    best_test_reward = test_reward
                print("%d: test reward=%.2f, steps=%.2f, best_reward=%.2f" % (
                    step_idx, test_reward, test_steps, best_test_reward))
