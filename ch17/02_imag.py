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
from torch.autograd import Variable

from lib import common, i2a


LEARNING_RATE = 5e-4
NUM_ENVS = 16
BATCH_SIZE = 64
SAVE_EVERY_BATCH = 1000
OBS_WEIGHT = 10.0
REWARD_WEIGHT = 1.0

def get_obs_diff(prev_obs, cur_obs):
    prev = np.array(prev_obs)[-1]
    cur = np.array(cur_obs)[-1]
    prev = prev.astype(np.float32) / 255.0
    cur = cur.astype(np.float32) / 255.0
    return cur - prev


def iterate_batches(envs, net, cuda=False):
    act_selector = ptan.actions.ProbabilityActionSelector()
    mb_obs = np.zeros((BATCH_SIZE, ) + common.IMG_SHAPE, dtype=np.uint8)
    mb_probs = np.zeros((BATCH_SIZE, envs[0].action_space.n), dtype=np.float32)
    mb_obs_next = np.zeros((BATCH_SIZE, ) + i2a.EM_OUT_SHAPE, dtype=np.float32)
    mb_actions = np.zeros((BATCH_SIZE, ), dtype=np.int32)
    mb_rewards = np.zeros((BATCH_SIZE, ), dtype=np.float32)
    obs = [e.reset() for e in envs]
    total_reward = [0.0] * NUM_ENVS
    total_steps = [0] * NUM_ENVS
    batch_idx = 0
    done_rewards = []
    done_steps = []

    while True:
        obs_v = ptan.agent.default_states_preprocessor(obs, cuda=cuda)
        logits_v, values_v = net(obs_v)
        probs_v = F.softmax(logits_v)
        probs = probs_v.data.cpu().numpy()
        actions = act_selector(probs)

        for e_idx, e in enumerate(envs):
            o, r, done, _ = e.step(actions[e_idx])
            mb_obs[batch_idx] = obs[e_idx]
            mb_probs[batch_idx] = probs[e_idx]
            mb_obs_next[batch_idx] = get_obs_diff(obs[e_idx], o)
            mb_actions[batch_idx] = actions[e_idx]
            mb_rewards[batch_idx] = r

            total_reward[e_idx] += r
            total_steps[e_idx] += 1

            batch_idx = (batch_idx + 1) % BATCH_SIZE
            if batch_idx == 0:
                yield mb_obs, mb_probs, mb_obs_next, mb_actions, mb_rewards, done_rewards, done_steps
                done_rewards.clear()
                done_steps.clear()
            if done:
                o = e.reset()
                done_rewards.append(total_reward[e_idx])
                done_steps.append(total_steps[e_idx])
                total_reward[e_idx] = 0.0
                total_steps[e_idx] = 0
            obs[e_idx] = o

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-m", "--model", required=True, help="File with model to load")
    args = parser.parse_args()

    saves_path = os.path.join("saves", "02_env_" + args.name)
    os.makedirs(saves_path, exist_ok=True)

    envs = [common.make_env() for _ in range(NUM_ENVS)]
    writer = SummaryWriter(comment="-02_env_" + args.name)

    net = common.AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n)
    net_em = i2a.EnvironmentModel(envs[0].observation_space.shape, envs[0].action_space.n)
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    if args.cuda:
        net.cuda()
        net_em.cuda()
    print(net_em)
    optimizer = optim.Adam(net_em.parameters(), lr=LEARNING_RATE)

    step_idx = 0
    best_reward = None
    with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
        for mb_obs, mb_probs, mb_obs_next, mb_actions, mb_rewards, done_rewards, done_steps in iterate_batches(envs, net, cuda=args.cuda):
            if len(done_rewards) > 0:
                m_reward = np.mean(done_rewards)
                m_steps = np.mean(done_steps)
                print("%d: done %d episodes, mean reward=%.2f, steps=%.2f" % (
                    step_idx, len(done_rewards), m_reward, m_steps))
                tb_tracker.track("total_reward", m_reward, step_idx)
                tb_tracker.track("total_steps", m_steps, step_idx)

            obs_v = Variable(torch.from_numpy(mb_obs))
            probs_v = Variable(torch.from_numpy(mb_probs))
            obs_next_v = Variable(torch.from_numpy(mb_obs_next))
            actions_t = torch.LongTensor(mb_actions.tolist())
            rewards_v = Variable(torch.from_numpy(mb_rewards))
            if args.cuda:
                obs_v = obs_v.cuda()
                probs_v = probs_v.cuda()
                actions_t = actions_t.cuda()
                obs_next_v = obs_next_v.cuda()
                rewards_v = rewards_v.cuda()

            optimizer.zero_grad()
            out_obs_next_v, out_reward_v = net_em(obs_v.float()/255, actions_t)
            loss_obs_v = F.mse_loss(out_obs_next_v, obs_next_v)
            loss_rew_v = F.mse_loss(out_reward_v, rewards_v)
            loss_total_v = OBS_WEIGHT * loss_obs_v + REWARD_WEIGHT * loss_rew_v
            loss_total_v.backward()
            # imag_policy_logits_v = net_imag_policy(obs_v)
            # imag_policy_loss_v = -F.log_softmax(imag_policy_logits_v) * probs_v
            # imag_policy_loss_v = imag_policy_loss_v.sum(dim=1).mean()
            # imag_policy_loss_v.backward()
            optimizer.step()
            tb_tracker.track("loss_em_obs", loss_obs_v, step_idx)
            tb_tracker.track("loss_em_reward", loss_rew_v, step_idx)
            tb_tracker.track("loss_em_total", loss_total_v, step_idx)
#            tb_tracker.track("imag_policy_loss", imag_policy_loss_v, step_idx)

            step_idx += 1
            if step_idx % SAVE_EVERY_BATCH == 0:
                loss = loss_total_v.data.cpu().numpy()
                fname = os.path.join(saves_path, "em_%05d_%.4e.dat" % (step_idx, loss))
                torch.save(net_em.state_dict(), fname)
