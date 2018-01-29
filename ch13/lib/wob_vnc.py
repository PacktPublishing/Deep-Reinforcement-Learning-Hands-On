import gym
import numpy as np
from PIL import Image, ImageDraw

from universe.wrappers.experimental import SoftmaxClickMouse
from universe import vectorized


# observation speed. Shouldn't be more than 15, as
# rewarder thread on the server side will fall behind.
FPS = 5

# Area of interest
WIDTH = 160
HEIGHT = 210
X_OFS = 10
Y_OFS = 75

WOB_SHAPE = (3, HEIGHT, WIDTH)


def remotes_url(port_ofs=0, hostname='localhost', count=8):
    hosts = ["%s:%d+%d" % (hostname, 5900 + ofs, 15900 + ofs) for ofs in range(port_ofs, port_ofs+count)]
    return "vnc://" + ",".join(hosts)


def configure(env, remotes, fps=FPS):
    env.configure(remotes=remotes, fps=fps, vnc_kwargs={
        'encoding': 'tight', 'compress_level': 0,
        'fine_quality_level': 100, 'subsample_level': 0
    })


class MiniWoBCropper(vectorized.ObservationWrapper):
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


def save_obs(obs, file_name, action=None, action_step_pix=10, action_y_ofs=50, transpose=True):
    """
    Save observation from the WoB
    :param obs: single observation (3d)
    :param file_name: image file to save
    :param action: action index to show on image
    :param action_step_pix: discrete step, taken from SoftmaxClickMouse
    :param action_y_ofs: y ofs, taken from SoftmaxClickMouse
    :return:
    """
    if transpose:
        obs = np.transpose(obs, (1, 2, 0))
    img = Image.fromarray(obs)
    if action is not None:
        draw = ImageDraw.Draw(img)
        if isinstance(action, tuple):
            x_ofs, y_ofs = action
            x_ofs -= X_OFS
            y_ofs -= Y_OFS
        else:
            y_ofs = action_y_ofs + (action % 16) * action_step_pix
            x_ofs = (action // 16) * action_step_pix
        half_step = action_step_pix//2
        draw.ellipse((x_ofs-half_step, y_ofs-half_step, x_ofs+half_step, y_ofs+half_step),
                     (0, 0, 255, 128))
    img.save(file_name)


class MiniWoBPeeker(vectorized.Wrapper):
    """
    Saves series of images with actions with a specifed prefix. Passes everything
    unchanged. Supposed to be inserted between SoftMaxClicker and MiniWoBCropper.
    """
    def __init__(self, env, img_prefix):
        super(MiniWoBPeeker, self).__init__(env)
        self.img_prefix = img_prefix
        self.episodes = None
        self.steps = None
        self.img_stack = None
        e = env
        while e is not None and not isinstance(e, SoftmaxClickMouse):
            e = e.env
        assert e is not None
        self.softmax_env = e

    def _reset(self):
        res = self.env.reset()
        if self.episodes is None:
            self.episodes = [0] * len(res)
        if self.steps is None:
            self.steps = [0] * len(res)
        self.img_stack = [None] * len(res)
        return res

    def _step(self, action_n):
        for img_item, action in zip(self.img_stack, action_n):
            if img_item is None:
                continue
            img, fname = img_item
            action_coords = self.softmax_env._points[action]
            save_obs(img, fname, action_coords, transpose=False)

        observation_n, reward_n, done_n, info = self.env.step(action_n)
        for idx, (obs, reward, done, action) in enumerate(zip(observation_n, reward_n, done_n, action_n)):
            if obs is not None and not done:
                fname = "%s_env%d_ep%03d_st%03d_rw%.2f_d%d.png" % (
                    self.img_prefix, idx, self.episodes[idx], self.steps[idx], reward, int(done)
                )
                img = obs['vision']
                img = img[Y_OFS:Y_OFS+HEIGHT, X_OFS:X_OFS+WIDTH*2, :]
                self.img_stack[idx] = (img, fname)
            else:
                self.img_stack[idx] = None
            if done:
                self.episodes[idx] += 1
                self.steps[idx] = 0
            else:
                self.steps[idx] += 1
        return observation_n, reward_n, done_n, info
