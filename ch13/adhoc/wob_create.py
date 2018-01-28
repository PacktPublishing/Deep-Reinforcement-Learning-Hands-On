#!/usr/bin/env python3
import gym
import universe
import time

from PIL import Image


if __name__ == "__main__":
    env = gym.make("wob.mini.ClickDialog-v0")

    env.configure(remotes=1, fps=5, vnc_kwargs={
        'encoding': 'tight', 'compress_level': 0,
        'fine_quality_level': 100, 'subsample_level': 0
    })
    obs = env.reset()

    while obs[0] is None:
        a = env.action_space.sample()
        obs, reward, is_done, info = env.step([a])
        print("Env is still resetting...")
        time.sleep(1)

    print(obs[0].keys())
    im = Image.fromarray(obs[0]['vision'])
    im.save("image.png")
    env.close()
