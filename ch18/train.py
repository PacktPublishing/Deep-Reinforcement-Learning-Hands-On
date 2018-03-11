#!/usr/bin/env python3
import os
import argparse
import collections
import numpy as np

from lib import game, model, mcts

from tensorboardX import SummaryWriter
import torch.optim as optim


MCTS_SEARCHES = 20
REPLAY_BUFFER = 1000


def play_game(replay_buffer, net1, net2, cuda=False):
    """
    Play one single game, memorizing transitions into the replay buffer
    :param replay_buffer: queue with (state, probs, values), if None, nothing is stored
    :param net1: player1
    :param net2: player2
    :return: value for the game in respect to player1 (+1 if p1 won, -1 if lost, 0 if draw)
    """
    assert isinstance(replay_buffer, (collections.deque, type(None)))
    state = game.INITIAL_STATE
    nets = [net1, net2]
    cur_player = np.random.choice(2)
    while True:
        mcts_store.search_batch(MCTS_SEARCHES, state, cur_player, nets[cur_player], cuda=cuda)
        probs, values = mcts_store.get_policy_value(state)
        if replay_buffer is not None:
            replay_buffer.append((state, probs, values))
        action = np.random.choice(game.GAME_COLS, p=probs)
        if action not in game.possible_moves(state):
            print("Impossible action selected")
        state, won = game.move(state, action, cur_player)
        if won:
            return 1.0 if cur_player == 0 else -1
        cur_player = 1-cur_player
        # check the draw case
        if len(game.possible_moves(state)) == 0:
            return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    args = parser.parse_args()

    saves_path = os.path.join("saves", args.name)
    os.makedirs(saves_path, exist_ok=True)
    writer = SummaryWriter(comment=args.name)

    net = model.Net(input_shape=model.OBS_SHAPE, actions_n=game.GAME_COLS)
    if args.cuda:
        net.cuda()
    best_net = net
    print(net)

    mcts_store = mcts.MCTS()
    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER)

    for game_idx in range(50):
        r = play_game(replay_buffer, net, net, cuda=args.cuda)
        print(r)
        print("MCTS=%d, replay=%d" % (len(mcts_store), len(replay_buffer)))


    pass
