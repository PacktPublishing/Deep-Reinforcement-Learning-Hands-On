#!/usr/bin/env python3
import numpy as np
import argparse

from lib import game, model

import torch
import torch.nn.functional as F


def play_game(net1, net2):
    cur_player = 0
    state = game.INITIAL_STATE
    nets = [net1, net2]

    while True:
        state_list = game.decode_binary(state)
        batch_v = model.state_lists_to_batch([state_list], [cur_player])
        logits_v, _ = nets[cur_player](batch_v)
        probs_v = F.softmax(logits_v)
        probs = probs_v[0].data.cpu().numpy()
        while True:
            action = np.random.choice(game.GAME_COLS, p=probs)
            if action in game.possible_moves(state):
                break
        state, won = game.move(state, action, cur_player)
        if won:
            return 1.0 if cur_player == 0 else -1.0
        # check for the draw state
        if len(game.possible_moves(state)) == 0:
            return 0.0
        cur_player = 1 - cur_player


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs='+', help="The list of models (at least 2) to play against each other")
    parser.add_argument("-r", "--rounds", type=int, default=2, help="Count of rounds to perform for every pair")
    args = parser.parse_args()

    nets = []
    for fname in args.models:
        net = model.Net(model.OBS_SHAPE, game.GAME_COLS)
        net.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
        nets.append((fname, net))

    table = np.zeros((len(nets), len(nets)))

    for idx1, n1 in enumerate(nets):
        for idx2, n2 in enumerate(nets):
            if idx1 == idx2:
                continue
            score = 0.0
            for _ in range(args.rounds):
                r = play_game(n1[1], n2[1])
                print(r)
                score += r
            print("%s vs %s -> %.1f" % (n1[0], n2[0], score))
            table[idx1][idx2] += score
            table[idx2][idx1] -= score

    pass
