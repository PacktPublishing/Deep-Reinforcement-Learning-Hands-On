#!/usr/bin/env python3
import sys
sys.path.append("..")
import time
import gym
# this import is not being used, but is required to make Universe envs available!
import universe

from PIL import Image

from lib import wob_vnc

# Docker image manual start:
# docker run -p 5900:5900 -p 15900:15900 --privileged --ipc host --cap-add SYS_ADMIN quay.io/openai/universe.world-of-bits:0.20.0


if __name__ == "__main__":
    env = gym.make("wob.mini.BisectAngle-v0")
    env = wob_vnc.MiniWoBCropper(env)

#    env.configure(remotes=1)   # local mode
    env.configure(remotes='vnc://gpu:5900+15900')
    print(env)
    obs = env.reset()
    saved = False

    while True:
        time.sleep(1)
        a = env.action_space.sample()
        obs, reward, is_done, info = env.step([a])
        if obs[0] is None:
            print("Env is still resetting...")
            continue
        print("Sampled action: ", a)
        print("Response are:")
        print("Observation", obs[0].shape)
        print("Reward", reward)
        print("Is done", is_done)
        print("Info", info)

        if not saved:
            im = Image.fromarray(obs[0])
            im.save("image-cropped.png")
            saved = True

    env.close()
    pass
