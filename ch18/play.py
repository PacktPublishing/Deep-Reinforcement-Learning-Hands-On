#!/usr/bin/env python3
import sys
import argparse

from lib import game, model

import torch


MCTS_SEARCHES = 3
MCTS_BATCH_SIZE = 30


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
            if idx1 == idx2:
                continue
            wins, losses, draws = 0, 0, 0
            for _ in range(args.rounds):
                r, _ = model.play_game(mcts_store=None, replay_buffer=None, net1=n1[1], net2=n2[1], steps_before_tau_0=0,
                                    mcts_searches=MCTS_SEARCHES, mcts_batch_size=MCTS_BATCH_SIZE)
                if r > 0.5:
                    wins += 1
                elif r < -0.5:
                    losses += 1
                else:
                    draws += 1
            name_1, name_2 = n1[0], n2[0]
            print("%s vs %s -> w=%d, l=%d, d=%d" % (name_1, name_2, wins, losses, draws))
            sys.stdout.flush()
            game.update_counts(total_agent, name_1, (wins, losses, draws))
            game.update_counts(total_agent, name_2, (losses, wins, draws))
            game.update_counts(total_pairs, (name_1, name_2), (wins, losses, draws))

    # leaderboard by total wins
    total_leaders = list(total_agent.items())
    total_leaders.sort(reverse=True, key=lambda p: p[1][0])

    print("Leaderboard:")
    for name, (wins, losses, draws) in total_leaders:
        print("%s: \t w=%d, l=%d, d=%d" % (name, wins, losses, draws))

    pass
