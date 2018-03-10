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
