import unittest

from lib import game


class TestBits(unittest.TestCase):
    def test_bits_to_int(self):
        v = game.bits_to_int([1, 0, 0])
        self.assertEqual(v, 4)
        v = game.bits_to_int([])
        self.assertEqual(v, 0)

    def test_int_to_bits(self):
        v = game.int_to_bits(1, bits=1)
        self.assertEqual(v, [1])
        v = game.int_to_bits(1, bits=5)
        self.assertEqual(v, [0, 0, 0, 0, 1])
        v = game.int_to_bits(5, bits=7)
        self.assertEqual(v, [0, 0, 0, 0, 1, 0, 1])


class TestGameEncoding(unittest.TestCase):
    def test_simple_encode(self):
        e = game.encode_lists([[]]*7)
        self.assertEqual(e, 0b000000000000000000000000000000000000000000110110110110110110110)
        e = game.encode_lists([[1]*6]*7)
        self.assertEqual(e, 0b111111111111111111111111111111111111111111000000000000000000000)
        e = game.encode_lists([[0]*6]*7)
        self.assertEqual(e, 0)

    def test_simple_decode(self):
        g = game.decode_binary(0b000000000000000000000000000000000000000000110110110110110110110)
        self.assertEqual(g, [[]]*7)
        g = game.decode_binary(0b111111111111111111111111111111111111111111000000000000000000000)
        self.assertEqual(g, [[1]*6]*7)
        g = game.decode_binary(0)
        self.assertEqual(g, [[0]*6]*7)


class TestMoveFunctions(unittest.TestCase):
    def test_possible_moves(self):
        r = game.possible_moves(0)
        self.assertEqual(r, [])
        r = game.possible_moves(0b111111111111111111111111111111111111111111000000000000000000000)
        self.assertEqual(r, [])
        r = game.possible_moves(0b000000000000000000000000000000000000000000110110110110110110110)
        self.assertEqual(r, [0, 1, 2, 3, 4, 5, 6])

    def test_move_vertical_win(self):
        f = game.encode_lists([[]]*7)

        f, won = game.move(f, 0, 1)
        self.assertFalse(won)
        l = game.decode_binary(f)
        self.assertEqual(l, [[1]] + [[]]*6)

        f, won = game.move(f, 0, 1)
        self.assertFalse(won)
        l = game.decode_binary(f)
        self.assertEqual(l, [[1, 1]] + [[]]*6)

        f, won = game.move(f, 0, 1)
        self.assertFalse(won)
        l = game.decode_binary(f)
        self.assertEqual(l, [[1, 1, 1]] + [[]]*6)

        f, won = game.move(f, 0, 1)
        self.assertTrue(won)
        l = game.decode_binary(f)
        self.assertEqual(l, [[1, 1, 1, 1]] + [[]]*6)

    def test_move_horizontal_win(self):
        f = game.encode_lists([[]]*7)

        f, won = game.move(f, 0, 1)
        self.assertFalse(won)
        l = game.decode_binary(f)
        self.assertEqual(l, [[1]] + [[]]*6)

        f, won = game.move(f, 1, 1)
        self.assertFalse(won)
        l = game.decode_binary(f)
        self.assertEqual(l, [[1], [1]] + [[]]*5)

        f, won = game.move(f, 3, 1)
        self.assertFalse(won)
        l = game.decode_binary(f)
        self.assertEqual(l, [[1], [1], [], [1], [], [], []])

        f, won = game.move(f, 2, 1)
        self.assertTrue(won)
        l = game.decode_binary(f)
        self.assertEqual(l, [[1], [1], [1], [1], [], [], []])

    def test_move_diags(self):
        f = game.encode_lists([
            [0, 0, 0, 1],
            [0, 0, 1],
            [0],
            [1],
            [], [], []
        ])
        _, won = game.move(f, 2, 1)
        self.assertTrue(won)
        _, won = game.move(f, 2, 0)
        self.assertFalse(won)

        f = game.encode_lists([
            [],
            [0, 1],
            [0, 0, 1],
            [1, 0, 0, 1],
            [], [], []
        ])
        _, won = game.move(f, 0, 1)
        self.assertTrue(won)
        _, won = game.move(f, 0, 0)
        self.assertFalse(won)

    def test_tricky(self):
        f = game.encode_lists([
            [0, 1, 1],
            [1, 0],
            [0, 1],
            [0, 0, 1],
            [0, 0],
            [1, 1, 1, 0],
            []
        ])
        s, won = game.move(f, 4, 0)
        self.assertTrue(won)
        self.assertEqual(s, 3531389463375529686)

pass
