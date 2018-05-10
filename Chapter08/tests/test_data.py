from unittest import TestCase
import numpy as np
from lib import data


class TestMisc(TestCase):
    def test_read_csv(self):
        prices = data.read_csv("data/YNDX_160101_161231.csv")
        self.assertIsInstance(prices, data.Prices)

    def test_prices_to_relative(self):
        t = data.Prices(open=np.array([1.0]),
                        high=np.array([3.0]),
                        low=np.array([0.5]),
                        close=np.array([2.0]),
                        volume=np.array([10]))
        rel = data.prices_to_relative(t)
        np.testing.assert_equal(rel.open,  t.open)
        np.testing.assert_equal(rel.volume,  t.volume)
        np.testing.assert_equal(rel.high,  np.array([2.0]))  # 200% growth
        np.testing.assert_equal(rel.low,   np.array([-.5]))  # 50% fall
        np.testing.assert_equal(rel.close, np.array([1.0]))  # 100% growth

    def test_price_files(self):
        files = data.price_files("data")
        self.assertTrue(len(files) > 0)

