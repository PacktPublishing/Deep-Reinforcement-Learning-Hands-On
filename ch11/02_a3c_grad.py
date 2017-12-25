#!/usr/bin/env python3
import gym
import ptan
import numpy as np
import argparse
import collections
from tensorboardX import SummaryWriter

import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.autograd import Variable

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01

REWARD_STEPS = 4
CLIP_GRAD = 0.1

PROCESSES_COUNT = 1
NUM_ENVS = 4

GRAD_BATCH = 10
TRAIN_BATCH = 16

if True:
    ENV_NAME = "PongNoFrameskip-v4"
    NAME = 'pong'
    REWARD_BOUND = 18
else:
    ENV_NAME = "BreakoutNoFrameskip-v4"
    NAME = "breakout"
    REWARD_BOUND = 400


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))

TotalReward = collections.namedtuple('TotalReward', field_names='reward')


class CachingA2CAgent(ptan.agent.BaseAgent):
    def __init__(self, model, cuda=False, preprocessor=ptan.agent.default_states_preprocessor):
        self.model = model
        self.cuda = cuda
        self.cache = {}
        self.action_selector = ptan.actions.ProbabilityActionSelector()
        self.preprocessor = preprocessor

    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * states.shape[0]
        if self.preprocessor is not None:
            prep_states = self.preprocessor(states)
        v = Variable(torch.from_numpy(prep_states))
        if self.cuda:
            v = v.cuda()
        probs_v, values_v = self.model(v)
        probs_v = F.softmax(probs_v)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)

        for idx, state in enumerate(states):
            self.cache[id(state)] = (probs_v[idx], values_v[idx])

        return np.array(actions), agent_states

    def pop(self, state):
        return self.cache.pop(id(state), None)

    def query_value(self, state):
        v = self.cache.get(id(state), None)
        if v is None:
            return None
        return v[1]


def data_func(net, cuda, train_queue, batch_size=GRAD_BATCH):
    envs = [make_env() for _ in range(NUM_ENVS)]

    tgt_net = ptan.agent.TargetNet(net)
    optimizer = optim.Adam(tgt_net.target_model.parameters())
    agent = CachingA2CAgent(tgt_net.target_model, cuda=cuda)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    batch_values = []
    batch_logits = []
    batch_ref_values = []
    batch_actions = []

    for exp in exp_source:
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            train_queue.put(TotalReward(reward=np.mean(new_rewards)))

        tgt_net.sync()

        s = agent.pop(exp.state)
        if s is None:
            print("Warning! No value for state")
            continue
        logits_v, value_v = s
        val_ref = exp.reward
        if exp.last_state is not None:
            value_last_v = agent.query_value(exp.last_state)
            val_ref += value_last_v.data.cpu().numpy()[0] * (GAMMA ** REWARD_STEPS)

        batch_ref_values.append(val_ref)
        batch_values.append(value_v)
        batch_logits.append(logits_v)
        batch_actions.append(int(exp.action))

        if len(batch_values) < batch_size:
            continue

        optimizer.zero_grad()
        vals_ref_v = Variable(torch.FloatTensor(batch_ref_values).unsqueeze(-1))
        logits_v = torch.stack(batch_logits)
        values_v = torch.stack(batch_values)
        actions_t = torch.LongTensor([batch_actions])
        if cuda:
            vals_ref_v = vals_ref_v.cuda()
            actions_t = actions_t.cuda()

        loss_value_v = F.mse_loss(values_v, vals_ref_v)

        log_prob_v = F.log_softmax(logits_v)
        adv_v = vals_ref_v - values_v.detach()
        log_prob_actions_v = adv_v * log_prob_v[range(batch_size), actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v)
        entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()
        loss_v = loss_value_v + loss_policy_v + entropy_loss_v
        loss_v.backward(retain_graph=True)

        # gather gradients
        grads = [param.grad for param in tgt_net.target_model.parameters()]
        train_queue.put(grads)

        batch_ref_values.clear()
        batch_values.clear()
        batch_logits.clear()
        batch_actions.clear()

    train_queue.put(None)


def unpack_batch(batch, net, cuda=False):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))
    states_v = Variable(torch.from_numpy(np.array(states, copy=False)))
    actions_t = torch.LongTensor(actions)
    if cuda:
        states_v = states_v.cuda()
        actions_t = actions_t.cuda()

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = Variable(torch.from_numpy(np.array(last_states, copy=False)), volatile=True)
        if cuda:
            last_states_v = last_states_v.cuda()
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_vals_np

    ref_vals_v = Variable(torch.from_numpy(rewards_np))
    if cuda:
        ref_vals_v = ref_vals_v.cuda()

    return states_v, actions_t, ref_vals_v


if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    writer = SummaryWriter(comment="-a3c-grad_" + NAME + "_" + args.name)

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

    batch = []
    step_idx = 0

    with common.RewardTracker(writer, stop_reward=REWARD_BOUND) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                train_entry = train_queue.get()
                if train_entry is None:
                    break
                if isinstance(train_entry, TotalReward):
                    if tracker.reward(train_entry.reward, step_idx):
                        break
                    continue

                step_idx += GRAD_BATCH

                if step_idx % TRAIN_BATCH == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                for param, grad in zip(net.parameters(), train_entry):
                    if grad is None:
                        continue
                    if param.grad is None:
                        param.grad = grad
                    else:
                        param.grad += grad
