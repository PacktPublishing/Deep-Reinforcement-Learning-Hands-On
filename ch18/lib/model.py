import collections
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from lib import game, mcts


OBS_SHAPE = (2, game.GAME_ROWS, game.GAME_COLS)
NUM_FILTERS = 64


class Net(nn.Module):
    def __init__(self, input_shape, actions_n):
        super(Net, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv2d(input_shape[0], NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )

        # layers with residual
        self.conv_1 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )

        body_out_shape = (NUM_FILTERS, ) + input_shape[1:]

        # value head
        self.conv_val = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
        conv_val_size = self._get_conv_val_size(body_out_shape)
        self.value = nn.Sequential(
            nn.Linear(conv_val_size, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
            nn.Tanh()
        )

        # policy head
        self.conv_policy = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU()
        )
        conv_policy_size = self._get_conv_policy_size(body_out_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_policy_size, actions_n)
        )

    def _get_conv_val_size(self, shape):
        o = self.conv_val(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def _get_conv_policy_size(self, shape):
        o = self.conv_policy(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        v = self.conv_in(x)
        v = v + self.conv_1(v)
        v = v + self.conv_2(v)
        v = v + self.conv_3(v)
        v = v + self.conv_4(v)
        v = v + self.conv_5(v)
        val = self.conv_val(v)
        val = self.value(val.view(batch_size, -1))
        pol = self.conv_policy(v)
        pol = self.policy(pol.view(batch_size, -1))
        return pol, val


def _encode_list_state(dest_np, state_list, who_move):
    """
    In-place encodes list state into the zero numpy array
    :param dest_np: dest array, expected to be zero
    :param state_list: state of the game in the list form
    :param who_move: player index (game.PLAYER_WHITE or game.PLAYER_BLACK) who to move
    """
    assert dest_np.shape == OBS_SHAPE

    for col_idx, col in enumerate(state_list):
        for rev_row_idx, cell in enumerate(col):
            row_idx = game.GAME_ROWS - rev_row_idx - 1
            if cell == who_move:
                dest_np[0, row_idx, col_idx] = 1.0
            else:
                dest_np[1, row_idx, col_idx] = 1.0


def state_lists_to_batch(state_lists, who_moves_lists, cuda=False):
    """
    Convert list of list states to batch for network
    :param state_lists: list of 'list states'
    :param who_moves_lists: list of player index who moves
    :return Variable with observations
    """
    assert isinstance(state_lists, list)
    batch_size = len(state_lists)
    batch = np.zeros((batch_size,) + OBS_SHAPE, dtype=np.float32)
    for idx, (state, who_move) in enumerate(zip(state_lists, who_moves_lists)):
        _encode_list_state(batch[idx], state, who_move)
    batch_v = Variable(torch.from_numpy(batch))
    if cuda:
        batch_v = batch_v.cuda()
    return batch_v


# def play_game(net1, net2, cuda=False):
#     cur_player = 0
#     state = game.INITIAL_STATE
#     nets = [net1, net2]
#
#     while True:
#         state_list = game.decode_binary(state)
#         batch_v = state_lists_to_batch([state_list], [cur_player], cuda)
#         logits_v, _ = nets[cur_player](batch_v)
#         probs_v = F.softmax(logits_v)
#         probs = probs_v[0].data.cpu().numpy()
#         while True:
#             action = np.random.choice(game.GAME_COLS, p=probs)
#             if action in game.possible_moves(state):
#                 break
#         state, won = game.move(state, action, cur_player)
#         if won:
#             return 1.0 if cur_player == 0 else -1.0
#         # check for the draw state
#         if len(game.possible_moves(state)) == 0:
#             return 0.0
#         cur_player = 1 - cur_player
#


def play_game(mcts_store, replay_buffer, net1, net2, steps_before_tau_0, mcts_searches, mcts_batch_size,
              net1_plays_first=None, cuda=False):
    """
    Play one single game, memorizing transitions into the replay buffer
    :param replay_buffer: queue with (state, probs, values), if None, nothing is stored
    :param net1: player1
    :param net2: player2
    :return: value for the game in respect to player1 (+1 if p1 won, -1 if lost, 0 if draw)
    """
    assert isinstance(replay_buffer, (collections.deque, type(None)))
    assert isinstance(mcts_store, (mcts.MCTS, type(None)))
    assert isinstance(net1, Net)
    assert isinstance(net2, Net)
    assert isinstance(steps_before_tau_0, int) and steps_before_tau_0 >= 0
    assert isinstance(mcts_searches, int) and mcts_searches > 0
    assert isinstance(mcts_batch_size, int) and mcts_batch_size > 0

    if mcts_store is None:
        mcts_store = mcts.MCTS()

    state = game.INITIAL_STATE
    nets = [net1, net2]
    if net1_plays_first is None:
        cur_player = np.random.choice(2)
    else:
        cur_player = 0 if net1_plays_first else 1
    step = 0
    result = None
    tau = 1 if steps_before_tau_0 > 0 else 0
    while result is None:
        mcts_store.search_batch(mcts_searches, mcts_batch_size, state, cur_player, nets[cur_player], cuda=cuda)
        probs, values = mcts_store.get_policy_value(state, tau=tau)
        if replay_buffer is not None:
            replay_buffer.append((state, cur_player, probs, max(values)))
        action = np.random.choice(game.GAME_COLS, p=probs)
        if action not in game.possible_moves(state):
            print("Impossible action selected")
        state, won = game.move(state, action, cur_player)
        if won:
            result = 1.0 if cur_player == 0 else -1
        cur_player = 1-cur_player
        # check the draw case
        if len(game.possible_moves(state)) == 0:
            result = 0
        step += 1
        if step >= steps_before_tau_0:
            tau = 0
    return result, step
