import gym
import numpy as np

# Area of interest
WIDTH = 160
HEIGHT = 210
X_OFS = 10
Y_OFS = 75

WOB_SHAPE = (3, HEIGHT, WIDTH)


def remotes_url(port_ofs=0, hostname='localhost', count=4):
    hosts = ["%s:%d+%d" % (hostname, 5900 + ofs, 15900 + ofs) for ofs in range(port_ofs, port_ofs+count)]
    return "vnc://" + ",".join(hosts)


def configure(env, remotes):
    env.configure(remotes=remotes, fps=5, vnc_kwargs={
        'encoding': 'tight', 'compress_level': 0,
        'fine_quality_level': 100, 'subsample_level': 0
    })


class MiniWoBCropper(gym.ObservationWrapper):
    """
    Crops the WoB area and converts the observation into PyTorch (C, H, W) format.
    """
    def _observation(self, observation_n):
        res = []
        for obs in observation_n:
            if obs is None:
                res.append(obs)
                continue
            img = obs['vision'][Y_OFS:Y_OFS+HEIGHT, X_OFS:X_OFS+WIDTH, :]
            res.append(np.transpose(img, (2, 0, 1)))
        return res

