#!/usr/bin/env python3
import sys
sys.path.append("..")
import time
import gym
import universe

from PIL import Image

from lib import wob_vnc


if __name__ == "__main__":
    env = gym.make("wob.mini.BisectAngle-v0")
    env = universe.wrappers.experimental.SoftmaxClickMouse(env)
    env = wob_vnc.MiniWoBCropper(env)

    env.configure(remotes='vnc://gpu:5900+15900')
    obs = env.reset()

    while True:
        a = env.action_space.sample()
        obs, reward, is_done, info = env.step([a])
        if obs[0] is None:
            print("Env is still resetting...")
            continue
        break

    for idx in range(100):
        time.sleep(1)
        a = env.action_space.sample()
        obs, reward, is_done, info = env.step([a])
        if obs[0] is None:
            print("Env is resetting...")
            continue
        print("Sampled action: ", a)
        print("Response are:")
        print("Observation", obs[0].shape)
        print("Reward", reward)
        print("Is done", is_done)
        print("Info", info)

        im = Image.fromarray(obs[0])
        im.save("wob_clicks/frame-%03d.png" % idx)

    env.close()
    pass
