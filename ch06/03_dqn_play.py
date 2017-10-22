import gym
import torch
import argparse

from lib import wrappers
from lib import dqn_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", required=True, help="Environment name to use")
    args = parser.parse_args()


