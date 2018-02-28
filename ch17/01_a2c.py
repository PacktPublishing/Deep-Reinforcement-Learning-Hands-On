#!/usr/bin/env python3
import os
import gym
import ptan
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as nn_utils
from torch.autograd import Variable

from lib import common


LEARNING_RATE = 7e-4
NUM_ENVS = 16

REWARD_BOUND = 400
TEST_EVERY_BATCH = 100


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


def iterate_train_batches(envs, net, cuda=False):
    act_selector = ptan.actions.ProbabilityActionSelector()
    obs = [e.reset() for e in envs]
    cur_dones = [False] * NUM_ENVS
    total_reward = [0.0] * NUM_ENVS
    total_steps = [0] * NUM_ENVS
    mb_obs = np.zeros((NUM_ENVS, common.REWARD_STEPS) + common.IMG_SHAPE, dtype=np.uint8)
    mb_rewards = np.zeros((NUM_ENVS, common.REWARD_STEPS), dtype=np.float32)
    mb_values = np.zeros((NUM_ENVS, common.REWARD_STEPS), dtype=np.float32)
    mb_dones = np.zeros((NUM_ENVS, common.REWARD_STEPS), dtype=np.bool)
    mb_actions = np.zeros((NUM_ENVS, common.REWARD_STEPS), dtype=np.int32)

    while True:
        done_rewards = []
        done_steps = []
        for n in range(common.REWARD_STEPS):
            obs_v = ptan.agent.default_states_preprocessor(obs)
            mb_obs[:, n] = obs_v.data.numpy()
            mb_dones[:,  n] = cur_dones
            if cuda:
                obs_v = obs_v.cuda()
            logits_v, values_v = net(obs_v)
            probs_v = F.softmax(logits_v)
            actions = act_selector(probs_v.data.cpu().numpy())
            mb_actions[:, n] = actions
            mb_values[:, n] = values_v.squeeze().data.cpu().numpy()
            for e_idx, e in enumerate(envs):
                o, r, done, _ = e.step(actions[e_idx])
                total_reward[e_idx] += r
                total_steps[e_idx] += 1
                if done:
                    o = e.reset()
                    done_rewards.append(total_reward[e_idx])
                    done_steps.append(total_steps[e_idx])
                    total_reward[e_idx] = 0.0
                    total_steps[e_idx] = 0
                obs[e_idx] = o
                mb_rewards[e_idx, n] = r
                cur_dones[e_idx] = done
        # obtain values for the last observation
        obs_v = ptan.agent.default_states_preprocessor(obs, cuda)
        _, values_v = net(obs_v)
        values_last = values_v.squeeze().data.cpu().numpy()
        # prepare before rollouts calculation
        mb_dones = np.roll(mb_dones, -1, axis=1)
        mb_dones[:, -1] = cur_dones

        for e_idx, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, values_last)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if not dones[-1]:
                rewards = discount_with_dones(rewards + [value], dones + [False], common.GAMMA)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, common.GAMMA)
            mb_rewards[e_idx] = rewards

        out_mb_obs = mb_obs.reshape((-1,) + common.IMG_SHAPE)
        out_mb_rewards = mb_rewards.flatten()
        out_mb_actions = mb_actions.flatten()
        out_mb_values = mb_values.flatten()
        yield out_mb_obs, out_mb_rewards, out_mb_actions, out_mb_values, np.array(done_rewards), np.array(done_steps)


def test_model(env, net, rounds=3, cuda=False):
    total_reward = 0.0
    total_steps = 0
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[1], cuda=cuda, apply_softmax=True)

    for _ in range(rounds):
        obs = env.reset()
        while True:
            action = agent([obs])[0][0]
            obs, r, done, _ = env.step(action)
            total_reward += r
            total_steps += 1
            if done:
                break
    return total_reward / rounds, total_steps / rounds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    saves_path = os.path.join("saves", "01_a2c_" + args.name)
    os.makedirs(saves_path, exist_ok=True)

    make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("BreakoutNoFrameskip-v4"))
    envs = [make_env() for _ in range(NUM_ENVS)]
    test_env = make_env()
    writer = SummaryWriter(comment="-breakout-a2c_" + args.name)

    net = common.AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n)
    if args.cuda:
        net.cuda()
    print(net)
    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, eps=1e-5)

    step_idx = 0
    best_reward = None
    best_test_reward = None
    with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
        for mb_obs, mb_rewards, mb_actions, mb_values, done_rewards, done_steps in iterate_train_batches(envs, net, cuda=args.cuda):
            if len(done_rewards) > 0:
                if best_reward is None:
                    best_reward = done_rewards.max()
                elif best_reward < done_rewards.max():
                    best_reward = done_rewards.max()
                tb_tracker.track("total_reward_max", best_reward, step_idx)
                tb_tracker.track("total_reward", done_rewards, step_idx)
                tb_tracker.track("total_steps", done_steps, step_idx)
                print("%d: done %d episodes, mean_reward=%.2f, best_reward=%.2f" % (
                    step_idx, len(done_rewards), done_rewards.mean(), best_reward))

            optimizer.zero_grad()
            mb_adv = mb_rewards - mb_values
            adv_v = Variable(torch.from_numpy(mb_adv))
            obs_v = Variable(torch.from_numpy(mb_obs))
            rewards_v = Variable(torch.from_numpy(mb_rewards))
            actions_t = torch.LongTensor(mb_actions.tolist())
            if args.cuda:
                adv_v = adv_v.cuda()
                obs_v = obs_v.cuda()
                rewards_v = rewards_v.cuda()
                actions_t = actions_t.cuda()
            logits_v, values_v = net(obs_v)
            log_prob_v = F.log_softmax(logits_v)
            log_prob_actions_v = adv_v * log_prob_v[range(len(mb_actions)), actions_t]

            loss_policy_v = -log_prob_actions_v.mean()
            loss_value_v = F.mse_loss(values_v, rewards_v)

            prob_v = F.softmax(logits_v)
            entropy_loss_v = (prob_v * log_prob_v).sum(dim=1).mean()
            loss_v = common.ENTROPY_BETA * entropy_loss_v + common.VALUE_LOSS_COEF * loss_value_v + loss_policy_v
            loss_v.backward()
            nn_utils.clip_grad_norm(net.parameters(), common.CLIP_GRAD)
            optimizer.step()

            tb_tracker.track("advantage", mb_adv, step_idx)
            tb_tracker.track("values", values_v, step_idx)
            tb_tracker.track("batch_rewards", rewards_v, step_idx)
            tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
            tb_tracker.track("loss_policy", loss_policy_v, step_idx)
            tb_tracker.track("loss_value", loss_value_v, step_idx)
            tb_tracker.track("loss_total", loss_v, step_idx)

            step_idx += 1

            if step_idx % TEST_EVERY_BATCH == 0:
                test_reward, test_steps = test_model(test_env, net, cuda=args.cuda)
                tb_tracker.track("test_reward", test_reward, step_idx)
                tb_tracker.track("test_steps", test_steps, step_idx)
                if best_test_reward is None or best_test_reward < test_reward:
                    if best_test_reward is not None:
                        fname = os.path.join(saves_path, "best_%08.3f_%d.dat" % (test_reward, step_idx))
                        torch.save(net.state_dict(), fname)
                    best_test_reward = test_reward
                print("%d: test reward=%.2f, steps=%.2f, best_reward=%.2f" % (
                    step_idx, test_reward, test_steps, best_test_reward))
