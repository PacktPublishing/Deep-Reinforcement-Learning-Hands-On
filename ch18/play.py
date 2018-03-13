#!/usr/bin/env python3
import sys
import argparse

from lib import game, model

import torch


def update_counts(counts_dict, key, counts):
    v = counts_dict.get(key, (0, 0, 0))
    res = (v[0] + counts[0], v[1] + counts[1], v[2] + counts[2])
    counts_dict[key] = res


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

    total_agent = {}
    total_pairs = {}

    for idx1, n1 in enumerate(nets):
        for idx2, n2 in enumerate(nets):
            wins, losses, draws = 0, 0, 0
            for _ in range(args.rounds):
                r = model.play_game(n1[1], n2[1])
                if r > 0.5:
                    wins += 1
                elif r < -0.5:
                    losses += 1
                else:
                    draws += 1
            name_1, name_2 = n1[0], n2[0]
            print("%s vs %s -> w=%d, l=%d, d=%d" % (name_1, name_2, wins, losses, draws))
            sys.stdout.flush()
            update_counts(total_agent, name_1, (wins, losses, draws))
            update_counts(total_agent, name_2, (losses, wins, draws))
            update_counts(total_pairs, (name_1, name_2), (wins, losses, draws))

    # leaderboard by total wins
    total_leaders = list(total_agent.items())
    total_leaders.sort(reverse=True, key=lambda p: p[1][0])

    print("Leaderboard:")
    for name, (wins, losses, draws) in total_leaders:
        print("%s: \t w=%d, l=%d, d=%d" % (name, wins, losses, draws))

    pass
