#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd())
sys.path.append("..")
import argparse
import random

from lib import vnc_demo, wob_vnc

DEFAULT_ENV = "wob.mini.ClickTest2-v0"


from universe.spaces import vnc_event
import gym
import universe


def test_mouse_coords():
    env = gym.make(DEFAULT_ENV)
    env = universe.wrappers.experimental.SoftmaxClickMouse(env)

    for _ in range(100):
        x = random.randint(0, 300)
        y = random.randint(0, 300)
        event = vnc_event.PointerEvent(x, y, 1)
        discr = env._action_to_discrete(event)
        discr2 = vnc_demo.mouse_to_action(x, y)
        assert discr == discr2
        pass



if __name__ == "__main__":
#    test_mouse_coords()
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--demo", required=True, help="Dir name to scan for demos")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV, help="Environment name to load, default=" + DEFAULT_ENV)
    parser.add_argument("-o", "--output", required=True, help="Output prefix to save images")
    args = parser.parse_args()

    demo = vnc_demo.load_demo(args.demo, args.env, read_text=True)
    print("Loaded %d demo samples" % len(demo))

    env = gym.make(args.env)
    env = universe.wrappers.experimental.SoftmaxClickMouse(env)

    for idx, (obs, action) in enumerate(demo):
        fname = "%s_%04d.png" % (args.output, idx)
        action_coords = env._points[action]
        img, text = obs
        wob_vnc.save_obs(img, fname, action_coords)
        print(fname, text)

    pass


