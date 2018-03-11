#!/usr/bin/env python3
from lib import game, model


if __name__ == "__main__":
    net = model.Net(input_shape=model.OBS_SHAPE, actions_n=game.GAME_COLS)
    print(net)
    pass
