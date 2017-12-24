#!/usr/bin/env python3
import gym
import ptan
import time
import numpy as np
import argparse
import collections
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.autograd import Variable

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1

PROCESSES_COUNT = 20

def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))


TrainEntry = collections.namedtuple('TrainEntry', field_names=['state', 'q', 'action'])


def sum_reward(exps):
    Q = 0
    for idx, e in enumerate(exps):
        Q += (GAMMA ** idx) * e.reward
    return Q


def data_func(net, cuda, train_queue):
    last_value = None
    exp_steps = collections.deque(maxlen=REWARD_STEPS)

    def policy_fun(x):
        nonlocal last_value
        policy_v, value_v = net(x)
        last_value = value_v.data.cpu().numpy()[0][0]
        return policy_v

    env = make_env()
    agent = ptan.agent.PolicyAgent(policy_fun, apply_softmax=True, cuda=cuda)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)

    for exp in exp_source:
        exp = exp[0]
        if exp.done:
            print("Done!")
            while exp_steps:
                Q = sum_reward(exp_steps)
                last_exp = exp_steps.popleft()
                entry = TrainEntry(state=last_exp.state, q=Q, action=last_exp.action)
                train_queue.put(entry)
            continue
        if len(exp_steps) == REWARD_STEPS:
            Q = sum_reward(exp_steps)
            last_exp = exp_steps.popleft()
            Q += GAMMA ** REWARD_STEPS * last_value
            entry = TrainEntry(state=last_exp.state, q=Q, action=last_exp.action)
            train_queue.put(entry)
        exp_steps.append(exp)

    train_queue.put(None)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    writer = SummaryWriter(comment="-a3c-data_pong_" + args.name)

    env = make_env()
    net = common.AtariA2C(env.observation_space.shape, env.action_space.n)
    if args.cuda:
        net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []
    for _ in range(PROCESSES_COUNT):
        data_proc = mp.Process(target=data_func, args=(net, args.cuda, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)

    batch_states = []
    batch_qs = []
    batch_actions = []
    batch_idx = 0

    with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
        while True:
            train_entry = train_queue.get()
            if train_entry is None:
                break
            batch_states.append(np.array(train_entry.state, copy=False))
            batch_qs.append(float(train_entry.q))
            batch_actions.append(int(train_entry.action))

            if len(batch_states) < BATCH_SIZE:
                continue

            batch_idx += 1
            states_v = Variable(torch.from_numpy(np.array(batch_states, copy=False)))
            qs_v = Variable(torch.FloatTensor(batch_qs))
            actions_t = torch.LongTensor(batch_actions)
            if args.cuda:
                states_v = states_v.cuda()
                qs_v = qs_v.cuda()
                actions_t = actions_t.cuda()

            optimizer.zero_grad()
            logits_v, value_v = net(states_v)

            loss_value_v = F.mse_loss(value_v, qs_v)

            log_prob_v = F.log_softmax(logits_v)
            adv_v = qs_v - value_v.detach()
            log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
            loss_policy_v = -log_prob_actions_v.mean()

            prob_v = F.softmax(logits_v)
            entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

            # calculate policy gradients only
            loss_policy_v.backward(retain_graph=True)
            grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                    for p in net.parameters()
                                    if p.grad is not None])

            # apply entropy and value gradients
            loss_v = entropy_loss_v + loss_value_v
            loss_v.backward()
            nn_utils.clip_grad_norm(net.parameters(), CLIP_GRAD)
            optimizer.step()
            # get full loss
            loss_v += loss_policy_v

            tb_tracker.track("advantage", adv_v, batch_idx)
            tb_tracker.track("values", value_v, batch_idx)
            tb_tracker.track("batch_rewards", qs_v, batch_idx)
            tb_tracker.track("loss_entropy", entropy_loss_v, batch_idx)
            tb_tracker.track("loss_policy", loss_policy_v, batch_idx)
            tb_tracker.track("loss_value", loss_value_v, batch_idx)
            tb_tracker.track("loss_total", loss_v, batch_idx)

            tb_tracker.track("grad_l2", np.sqrt(np.mean(np.square(grads))), batch_idx)
            tb_tracker.track("grad_max", np.max(np.abs(grads)), batch_idx)
            tb_tracker.track("grad_var", np.var(grads), batch_idx)

            batch_states.clear()
            batch_qs.clear()
            batch_actions.clear()

pass
