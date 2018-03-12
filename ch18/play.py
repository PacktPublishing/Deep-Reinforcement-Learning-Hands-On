#!/usr/bin/env python3
import numpy as np
import argparse

from lib import game, model

import torch


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
                r = model.play_game(n1[1], n2[1])
                print(r)
                score += r
            print("%s vs %s -> %.1f" % (n1[0], n2[0], score))
            table[idx1][idx2] += score
            table[idx2][idx1] -= score

    pass
