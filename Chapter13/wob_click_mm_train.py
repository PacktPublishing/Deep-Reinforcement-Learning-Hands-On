#!/usr/bin/env python3
import os
import gym
import random
import universe
import argparse
import numpy as np
from tensorboardX import SummaryWriter

from lib import wob_vnc, model_vnc, common, vnc_demo

import ptan

import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim


REMOTES_COUNT = 8
ENV_NAME = "wob.mini.ClickTab-v0"

GAMMA = 0.99
REWARD_STEPS = 2
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
ENTROPY_BETA = 0.001
CLIP_GRAD = 0.05

DEMO_PROB = 0.5
CUT_DEMO_PROB_FRAMES = 50000        # After how many frames, demo probability will be dropped to 1%

SAVES_DIR = "saves"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--cuda", default=False, action='store_true', help="CUDA mode")
    parser.add_argument("--port-ofs", type=int, default=0, help="Offset for container's ports, default=0")
    parser.add_argument("--env", default=ENV_NAME, help="Environment name to solve, default=" + ENV_NAME)
    parser.add_argument("--demo", help="Demo dir to load. Default=No demo")
    parser.add_argument("--host", default='localhost', help="Host with docker containers")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env_name = args.env
    if not env_name.startswith('wob.mini.'):
        env_name = "wob.mini." + env_name

    name = env_name.split('.')[-1] + "_" + args.name
    writer = SummaryWriter(comment="-wob_click_mm_" + name)
    saves_path = os.path.join(SAVES_DIR, name)
    os.makedirs(saves_path, exist_ok=True)

    demo_samples = None
    if args.demo:
        demo_samples = vnc_demo.load_demo(args.demo, env_name, read_text=True)
        if not demo_samples:
            demo_samples = None
        else:
            print("Loaded %d demo samples, will use them during training" % len(demo_samples))

    env = gym.make(env_name)
    env = universe.wrappers.experimental.SoftmaxClickMouse(env)
    env = wob_vnc.MiniWoBCropper(env, keep_text=True)
    wob_vnc.configure(env, wob_vnc.remotes_url(port_ofs=args.port_ofs, hostname=args.host, count=REMOTES_COUNT))

    net = model_vnc.ModelMultimodal(input_shape=wob_vnc.WOB_SHAPE, n_actions=env.action_space.n).to(device)
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    preprocessor = model_vnc.MultimodalPreprocessor(device=device)
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device=device,
                                   apply_softmax=True, preprocessor=preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        [env], agent, gamma=GAMMA, steps_count=REWARD_STEPS, vectorized=True)

    best_reward = None
    with common.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            batch = []
            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", np.mean(steps), step_idx)

                    mean_reward = tracker.reward(np.mean(rewards), step_idx)
                    if mean_reward is not None:
                        if best_reward is None or mean_reward > best_reward:
                            if best_reward is not None:
                                name = "best_%.3f_%d" % (mean_reward, step_idx)
                                fname = os.path.join(saves_path, name)
                                torch.save(net.state_dict(), fname + ".dat")
                                preprocessor.save(fname + ".pre")
                                print("Best reward updated: %.3f -> %.3f" % (best_reward, mean_reward))
                            best_reward = mean_reward
                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                if step_idx > CUT_DEMO_PROB_FRAMES:
                    DEMO_PROB = 0.01
                if demo_samples and random.random() < DEMO_PROB:
                    random.shuffle(demo_samples)
                    demo_batch = demo_samples[:BATCH_SIZE]
                    model_vnc.train_demo(net, optimizer, demo_batch, writer, step_idx,
                                         preprocessor=preprocessor,
                                         device=device)

                states_v, actions_t, vals_ref_v = \
                    common.unpack_batch(batch, net, last_val_gamma=GAMMA ** REWARD_STEPS,
                                        device=device, states_preprocessor=preprocessor)
                batch.clear()

                optimizer.zero_grad()
                logits_v, value_v = net(states_v)

                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                loss_v = entropy_loss_v + loss_value_v + loss_policy_v
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
                tb_tracker.track("dict_size", len(preprocessor), step_idx)

    pass
