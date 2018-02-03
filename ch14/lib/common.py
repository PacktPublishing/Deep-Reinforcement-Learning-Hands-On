import numpy as np
import torch
from torch.autograd import Variable

import ptan


def unpack_batch_a2c(batch, net, last_val_gamma, cuda=False):
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
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states, cuda=cuda)
    actions_v = Variable(torch.from_numpy(np.array(actions, dtype=np.float32)))
    if cuda:
        actions_v = actions_v.cuda()

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = ptan.agent.float32_preprocessor(last_states, cuda=cuda)
        last_vals_v = net(last_states_v)[2]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = Variable(torch.from_numpy(rewards_np))
    if cuda:
        ref_vals_v = ref_vals_v.cuda()

    return states_v, actions_v, ref_vals_v


def unpack_batch_ddqn(batch, cuda=False):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states, cuda=cuda)
    actions_v = ptan.agent.float32_preprocessor(actions, cuda=cuda)
    rewards_v = ptan.agent.float32_preprocessor(rewards, cuda=cuda)
    dones_t = torch.ByteTensor(dones)
    last_states_v = Variable(torch.from_numpy(np.array(last_states, dtype=np.float32)), volatile=True)
    if cuda:
        dones_t = dones_t.cuda()
        last_states_v = last_states_v.cuda()
    return states_v, actions_v, rewards_v, dones_t, last_states_v
