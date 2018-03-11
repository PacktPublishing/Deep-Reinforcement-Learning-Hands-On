#!/usr/bin/env python3
from lib import game, model, mcts


if __name__ == "__main__":
    net = model.Net(input_shape=model.OBS_SHAPE, actions_n=game.GAME_COLS)
    print(net)

    mcts_store = mcts.MCTS()

    s = game.encode_lists([[]]*7)
    cur_player = game.PLAYER_WHITE
    p = mcts_store.search(s, cur_player, net, tau=1)
    pass
