import gym
import collections

# Area of interest
WIDTH = 160
HEIGHT = 210
X_OFS = 10
Y_OFS = 75


class MiniWoBCropper(gym.ObservationWrapper):
    def _observation(self, observation_n):
        res = []
        for obs in observation_n:
            if obs is None:
                res.append(obs)
                continue
            res.append(obs['vision'][Y_OFS:Y_OFS+HEIGHT, X_OFS:X_OFS+WIDTH, :])
        return res

