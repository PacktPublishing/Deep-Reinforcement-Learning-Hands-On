from unittest import TestCase

from libbots import utils


class TestBLEU(TestCase):
    def test_iterate_n_grams(self):
        r = list(utils.iterate_n_grams([1, 2, 3], n=1))
        self.assertEqual(r, [1, 2, 3])
        r = list(utils.iterate_n_grams([1, 2, 3], n=2))
        self.assertEqual(r, [(1, 2), (2, 3)])

    def test_calc_bleu(self):
        r = utils.calc_bleu(["i", "appreciate", "you", "telling", "me", ".", "#END"],
                            ["you", "still", "have", "describe", "her", ".", "#END"])

        r = utils.calc_bleu(["james", "looks", "twice", "?", "#END"],
                            ["james", "dressed", "twice", "?", "#END"])
        self.assertNotAlmostEqual(r, 1.0)
        r = utils.calc_bleu([1, 2, 3], [1, 2, 3], max_n_grams=1)
        self.assertAlmostEqual(r, 1.0)
        # r = utils.calc_bleu(["the", "the", "the", "the", "the", "the", "the"],
        #                     ["the", "cat", "is", "on", "the", "mat"], max_n_grams=1)
        # self.assertAlmostEqual(r, 2/7)
        # r = utils.calc_bleu(["I", "always", "invariably", "perpetually", "do"],
        #                     ["I", "always", "do"], max_n_grams=2)
        # self.assertAlmostEqual(r, 0.425)
        # r = utils.calc_bleu([1, 2, 3], [1, 2], max_n_grams=2)
        # self.assertAlmostEqual(r, 0.5*(2/3 + 1/2))

        r = utils.calc_bleu(["really", "!", "#END"], ["really", "!", "#END"])
        self.assertAlmostEqual(r, 1.0)

        # r = utils.calc_bleu(["a"], ["a", "a"])
        # self.assertEqual(r, 0.0)
