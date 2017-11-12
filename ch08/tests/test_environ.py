import unittest
import numpy as np

from lib import data, environ


class TestEnv(unittest.TestCase):
    def test_simple(self):
        prices = data.load_relative("data/YNDX_160101_161231.csv")
        env = environ.StocksEnv({"YNDX": prices})
        s = env.reset()
        obs, reward, done, info = env.step(0)
        self.assertAlmostEqual(reward, 0.0, 6)


class TestStates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = data.Prices(open=np.array([1.0, 2.0, 3.0, 1.0]),
                        high=np.array([2.0, 3.0, 4.0, 2.0]),
                        low=np.array([0.0, 1.0, 2.0, 0.0]),
                        close=np.array([2.0, 3.0, 1.0, 2.0]))
        cls.prices = {"TST": data.prices_to_relative(p)}


    def test_basic(self):
        s = environ.State(bars_count=4, comission_perc=0.0, reset_on_close=False)
        self.assertEqual(len(s), 4*3+2)

    def test_reset(self):
        s = environ.State(bars_count=1, comission_perc=0.0, reset_on_close=False)
        s.reset(self.prices['TST'], offset=0)
        self.assertFalse(s.have_position)
        self.assertAlmostEqual(s._cur_close(), 2.0)
        r, done = s.step(environ.Actions.Skip)
        self.assertAlmostEqual(s._cur_close(), 3.0)
        self.assertAlmostEqual(r, 0.0)
        self.assertFalse(done)
        r, done = s.step(environ.Actions.Skip)
        self.assertAlmostEqual(s._cur_close(), 1.0)
        self.assertAlmostEqual(r, 0.0)
        self.assertFalse(done)
        r, done = s.step(environ.Actions.Skip)
        self.assertAlmostEqual(s._cur_close(), 2.0)
        self.assertAlmostEqual(r, 0.0)
        self.assertTrue(done)

    def test_reward(self):
        s = environ.State(bars_count=1, comission_perc=0.0, reset_on_close=False)
        s.reset(self.prices['TST'], offset=0)
        self.assertFalse(s.have_position)
        self.assertAlmostEqual(s._cur_close(), 2.0)
        r, done = s.step(environ.Actions.Buy)
        self.assertTrue(s.have_position)
        self.assertFalse(done)
        self.assertAlmostEqual(r, 0.0)
        self.assertAlmostEqual(s._cur_close(), 3.0)
        r, done = s.step(environ.Actions.Skip)
        self.assertFalse(done)
        self.assertAlmostEqual(r, -2.0)
        self.assertAlmostEqual(s._cur_close(), 1.0)
        r, done = s.step(environ.Actions.Skip)
        self.assertTrue(done)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(s._cur_close(), 2.0)

    def test_comission(self):
        s = environ.State(bars_count=1, comission_perc=1.0, reset_on_close=False)
        s.reset(self.prices['TST'], offset=0)
        self.assertFalse(s.have_position)
        self.assertAlmostEqual(s._cur_close(), 2.0)
        r, done = s.step(environ.Actions.Buy)
        self.assertTrue(s.have_position)
        self.assertFalse(done)
        self.assertAlmostEqual(r, -3.0/100.0)  # execution price is next bar's close, comission 1%
        self.assertAlmostEqual(s._cur_close(), 3.0)
