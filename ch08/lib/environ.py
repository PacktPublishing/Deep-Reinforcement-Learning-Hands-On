import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np

from . import data

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2
    KeepOpen = 3


class State:
    def __init__(self, bars_count, comission_perc, reset_on_close, reward_on_close=True):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(comission_perc, float)
        assert comission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.comission = comission_perc / 100.0
        self.reset_on_close = reset_on_close

    def reset(self, prices, offset):
        assert isinstance(prices, data.Prices)
        assert offset >= self.bars_count-1
        self._prices = prices
        self._offset = offset

    @property
    def shape(self):
        # [h, l, c] * bars
        return (3*self.bars_count, )

    def encode(self):
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count+1, 1):
            res[shift] = self._prices.high[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.low[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.close[self._offset + bar_idx]
            shift += 1
        return res

    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)

    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        if action == Actions.Buy:
            reward -= self.comission
        elif action == Actions.Close:
            reward -= self.comission
            done |= self.reset_on_close

        self._offset += 1
        done |= self._offset >= self._prices.close.shape[0]-1

        if action == Actions.KeepOpen:
            reward += self._prices.close[self._offset]

        return reward, done


class State1D(State):
    """
    State with shape suitable for 1D convolution
    """
    @property
    def shape(self):
        return (3, self.bars_count)

    def encode(self):
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count-1
        res[0] = self._prices.high[self._offset-ofs:self._offset+1]
        res[1] = self._prices.low[self._offset-ofs:self._offset+1]
        res[2] = self._prices.close[self._offset-ofs:self._offset+1]
        return res


class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT,
                 comission=DEFAULT_COMMISSION_PERC, reset_on_close=True, state_1d=False,
                 random_ofs_on_reset=True, reward_on_close=True):
        assert isinstance(prices, dict)
        self._prices = prices
        if state_1d:
            self._state = State1D(bars_count, comission, reset_on_close, reward_on_close=reward_on_close)
        else:
            self._state = State(bars_count, comission, reset_on_close, reward_on_close=reward_on_close)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape)
        self.random_ofs_on_reset = random_ofs_on_reset
        self._seed()

    def _reset(self):
        # make selection of the instrument and it's offset. Then reset the state
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(prices.high.shape[0]-bars*10) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.encode()

    def _step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": self._state._offset}
        return obs, reward, done, info

    def _render(self, mode='human', close=False):
        pass

    def _close(self):
        pass

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = {file: data.load_relative(file) for file in data.price_files(data_dir)}
        return StocksEnv(prices, **kwargs)
