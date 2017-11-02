import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_loss_dqn(batch, net, tgt_net, gamma, cuda=False):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = Variable(torch.from_numpy(states))
    next_states_v = Variable(torch.from_numpy(next_states), volatile=True)
    actions_v = Variable(torch.from_numpy(actions))
    rewards_v = Variable(torch.from_numpy(rewards))
    done_mask = torch.ByteTensor(dones)
    if cuda:
        states_v = states_v.cuda()
        next_states_v = next_states_v.cuda()
        actions_v = actions_v.cuda()
        rewards_v = rewards_v.cuda()
        done_mask = done_mask.cuda()

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values.volatile = False

    expected_state_action_values = next_state_values * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)



def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = rewards + (Vmin + atom * delta_z) * gamma
        b_j = 1e-6 + (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        l_mask = np.logical_and(l >= 0, l < n_atoms)
        u_mask = np.logical_and(u >= 0, u < n_atoms)
        proj_distr[l_mask, l[l_mask]] += next_distr[l_mask, atom] * ((u - b_j)[l_mask])
        proj_distr[u_mask, u[u_mask]] += next_distr[u_mask, atom] * ((b_j - l)[u_mask])
    if dones.any():
        proj_distr[dones] = 0.0
        # Warning: here we assume that our rewards at the end of the episode will be in Vmin...Vmax range
        b_j = 1e-6 + (rewards[dones] - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        proj_distr[dones, l] += u - b_j
        proj_distr[dones, u] += b_j - l
    return proj_distr
