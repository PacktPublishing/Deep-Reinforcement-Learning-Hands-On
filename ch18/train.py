#!/usr/bin/env python3
import numpy as np

from lib import game, model, mcts


MCTS_SEARCHES = 20


if __name__ == "__main__":
    net = model.Net(input_shape=model.OBS_SHAPE, actions_n=game.GAME_COLS)
    print(net)

    mcts_store = mcts.MCTS()

    for game_idx in range(5):
        print("Game %d stars" % game_idx)
        s = game.INITIAL_STATE
        cur_player = game.PLAYER_WHITE
        move_idx = 0

        while True:
            print("Move %d:" % move_idx)
            print("\n".join(game.render(s)))
            mcts_store.search_batch(MCTS_SEARCHES, s, cur_player, net)
            probs, values = mcts_store.get_policy_value(s)
            action = np.random.choice(game.GAME_COLS, p=probs)
            if action not in game.possible_moves(s):
                print("Impossible action selected")
            print("Action %d, probs %s" % (action, probs))
            new_s, won = game.move(s, action, cur_player)
            if won:
                print("Player %d won the game!" % cur_player)
                break
            cur_player = 1-cur_player
            s = new_s
            # check the draw case
            if len(game.possible_moves(s)) == 0:
                break
            move_idx += 1
    pass
