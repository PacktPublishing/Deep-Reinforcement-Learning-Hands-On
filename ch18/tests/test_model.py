import unittest

import numpy as np
from lib import game, model


class TestEncoding(unittest.TestCase):
    def test_encoding(self):
        s = [[0, 1, 0], [0], [1, 1, 1], [], [1], [], []]
        batch_v = model.state_lists_to_batch([s, s], [game.PLAYER_BLACK, game.PLAYER_WHITE])
        batch = batch_v.data.numpy()
        np.testing.assert_equal(batch, [
            # black player's view
            [
                # player
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 0, 0],
                ],
                # opponent
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0],
                ]
            ],
            # white player's view
            [
                # player
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0],
                ],
                # opponent
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 0, 0],
                ]
            ],
        ])


pass
