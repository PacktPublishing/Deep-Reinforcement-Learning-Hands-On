import gym
import gym.spaces
import enum
import numpy as np

from . import data

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION = 0.01


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class State:
    def __init__(self, bars_count):
        self.bars_count = bars_count

    def reset(self, prices, offset):
        assert isinstance(prices, data.Prices)
        assert offset >= self.bars_count
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    def __len__(self):
        # [h, l, c] * bars + position_flag + rel_profit (since open)
        return 3*self.bars_count + 1 + 1

    def encode(self):
        """
        Convert data to numpy array. Return None if there is no more data available
        Offset is not updated
        """
        if self.bars_count + self._offset > self._prices.close.shape[0]:
            return None
        res = np.ndarray(shape=(len(self), ), dtype=np.float32)
        shift = 0
        for bar_idx in range(self.bars_count):
            res[shift] = self._prices.high[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.low[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.close[self._offset + bar_idx]
            shift += 1
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            open = self._prices.open[self._offset-1]
            rel_close = self._prices.close[self._offset-1]
            close = open * (1.0 + rel_close)
            res[shift] = (close - self.open_price) / self.open_price
        shift += 1
        return res


class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT):
        assert isinstance(prices, dict)
        self._prices = prices
        self._state = State(bars_count)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self._state), ))

    def _reset(self):
        # make selection of the instrument and it's offset. Then reset the state
        self._instrument = np.random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        offset = np.random.choice(prices.high.shape[0]-bars*10) + bars
        self._state.reset(prices, offset)
        return self._state.encode()

    def _step(self, action_idx):
        action = Actions(action_idx)
        reward = 0.0


    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = {name: data.load_relative(file) for name, file in data.price_files(data_dir)}
        return StocksEnv(prices, **kwargs)
