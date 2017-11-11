import gym

from . import data


class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_dir):
        self._prices = {name: data.read_csv(file) for name, file in data.price_files(data_dir)}
        
