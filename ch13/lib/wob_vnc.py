import gym
import numpy as np
from PIL import Image, ImageDraw

# observation speed. Shouldn't be more than 15, as
# rewarder thread on the server side will fall behind.
FPS = 15

# Area of interest
WIDTH = 160
HEIGHT = 210
X_OFS = 10
Y_OFS = 75

WOB_SHAPE = (3, HEIGHT, WIDTH)


def remotes_url(port_ofs=0, hostname='localhost', count=8):
    hosts = ["%s:%d+%d" % (hostname, 5900 + ofs, 15900 + ofs) for ofs in range(port_ofs, port_ofs+count)]
    return "vnc://" + ",".join(hosts)


def configure(env, remotes):
    env.configure(remotes=remotes, fps=FPS, vnc_kwargs={
        'encoding': 'tight', 'compress_level': 0,
        'fine_quality_level': 100, 'subsample_level': 0
    })


class MiniWoBCropper(gym.ObservationWrapper):
    """
    Crops the WoB area and converts the observation into PyTorch (C, H, W) format.
    """
    def __init__(self, env, keep_text=False):
        super(MiniWoBCropper, self).__init__(env)
        self.keep_text = keep_text

    def _observation(self, observation_n):
        res = []
        for obs in observation_n:
            if obs is None:
                res.append(obs)
                continue
            img = obs['vision'][Y_OFS:Y_OFS+HEIGHT, X_OFS:X_OFS+WIDTH, :]
            img = np.transpose(img, (2, 0, 1))
            if self.keep_text:
                text = " ".join(map(lambda d: d.get('instruction', ''), obs.get('text', [{}])))
                res.append((img, text))
            else:
                res.append(img)
        return res


def save_obs(obs, file_name, action=None, action_step_pix=10, action_y_ofs=50):
    """
    Save observation from the WoB
    :param obs: single observation (3d)
    :param file_name: image file to save
    :param action: action index to show on image
    :param action_step_pix: discrete step, taken from SoftmaxClickMouse
    :param action_y_ofs: y ofs, taken from SoftmaxClickMouse
    :return:
    """
    img = Image.fromarray(np.transpose(obs, (1, 2, 0)))
    if action is not None:
        draw = ImageDraw.Draw(img)
        y_ofs = action_y_ofs + (action % 16) * action_step_pix
        x_ofs = (action // 16) * action_step_pix
        draw.ellipse((x_ofs, y_ofs, x_ofs+action_step_pix, y_ofs+action_step_pix),
                     (0, 0, 255, 128))
    img.save(file_name)
