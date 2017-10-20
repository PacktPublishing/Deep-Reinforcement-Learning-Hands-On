#!/usr/bin/env python3
import sys
import gym
import gym.spaces
import random
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from PIL import Image

import cv2
from collections import deque
from gym import spaces


GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

SUMMARY_EVERY_FRAME = 100


class ImageWrapper(gym.ObservationWrapper):
    TARGET_SIZE = 84

    def __init__(self, env):
        super(ImageWrapper, self).__init__(env)
        probe = np.zeros_like(env.observation_space.low, np.uint8)
        self.observation_space = gym.spaces.Box(0, 255, self._observation(probe).shape)

    def _observation(self, obs):
        img = Image.fromarray(obs)
        img = img.convert("YCbCr")
        img = img.resize((self.TARGET_SIZE, self.TARGET_SIZE))
        data = np.asarray(img.getdata(0), np.uint8).reshape(img.size)
        return np.expand_dims(data, 0)


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.uint8):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0))

    def _reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self._observation(self.env.reset())

    def _observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer



class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]))

    def _observation(self, observation):
        return np.moveaxis(observation, 2, 0)


# Wrappers from OpenAI baselines
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def _step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def _observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ClippedRewardsWrapper(gym.RewardWrapper):
    def _reward(self, reward):
        """Change all the positive rewards to 1, negative to -1 and keep zero."""
        return np.sign(reward)


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not belive how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=2)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k))

    def _reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def _observation(self, obs):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(obs).astype(np.float) / 255.0


def wrap_dqn(env):
    """Apply a common set of wrappers for Atari games."""
    assert 'NoFrameskip' in env.spec.id
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    # if 'FIRE' in env.unwrapped.get_action_meanings():
    #     env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = FrameStack(env, 4)
    env = ClippedRewardsWrapper(env)
    return env


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class Buffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, item):
        self.buffer.append(item)

    def sample_batch(self, batch_size):
        obses_t, actions, rewards, dones, obses_tp1 = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, len(self.buffer)-1)
            obs_t, action, reward, done, obs_tp1 = self.buffer[idx]

            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t, dtype=np.float32), \
               np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones), \
               np.array(obses_tp1, dtype=np.float32)


if __name__ == "__main__":
    env = gym.make("PongNoFrameskip-v4")
    env = ImageToPyTorch(ScaledFloatFrame(wrap_dqn(env)))
#    env = ScaledFloatFrame(ImageToPyTorch(FrameStack(ProcessFrame84(MaxAndSkipEnv(env)), 4)))

    net = DQN(env.observation_space.shape, env.action_space.n)
    tgt_net = DQN(env.observation_space.shape, env.action_space.n)

    net.cuda()
    tgt_net.cuda()

    buffer = Buffer(REPLAY_SIZE)
    epsilon = 1.0

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = [0.0]
    state = env.reset()
    frame_idx = 0
    ts_frame = 0
    ts = time.time()

    while True:
        frame_idx += 1
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_vals_t = net(Variable(torch.FloatTensor([state]), volatile=True).cuda())[0]
            q_vals = q_vals_t.data.cpu().numpy()
            action = np.argmax(q_vals)

        new_state, reward, done, _ = env.step(action)
        buffer.add((state, action, reward, done, new_state))
        state = new_state
        total_rewards[-1] += reward
        if done:
            state = env.reset()
            total_rewards.append(0.0)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), np.mean(total_rewards[-100:]), epsilon,
                speed
            ))
            sys.stdout.flush()

        if len(buffer) < REPLAY_START_SIZE:
            continue

        # train
        optimizer.zero_grad()
        obses_t, actions, rewards, dones, obses_tp1 = buffer.sample_batch(BATCH_SIZE)

        obses_t_v = Variable(torch.from_numpy(obses_t)).cuda()
        obses_tp1_v = Variable(torch.from_numpy(obses_tp1), volatile=True).cuda()
        actions_v = Variable(torch.from_numpy(actions)).cuda()
        rewards_v = Variable(torch.from_numpy(rewards)).cuda()
        done_mask = torch.ByteTensor(dones.astype(np.uint8)).cuda()

        state_action_values = net(obses_t_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = tgt_net(obses_tp1_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values.volatile = False

        expected_state_action_values = next_state_values * GAMMA + rewards_v
        loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)

        # for idx in range(BATCH_SIZE):
        #     R = rewards[idx]
        #     if not dones[idx]:
        #         R += GAMMA * np.max(q_vals_tp1[idx])
        #     loss_t += (q_vals_t[idx][actions[idx]] - R) ** 2
        # loss_t /= BATCH_SIZE
        loss_t.backward()
        optimizer.step()

        epsilon = max(0.02, 1.0 - frame_idx / 10 ** 5)

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
    pass
