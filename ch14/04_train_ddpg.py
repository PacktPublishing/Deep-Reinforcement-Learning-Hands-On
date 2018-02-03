#!/usr/bin/env python3
import os
import ptan
import gym
import pybullet_envs
import argparse
from tensorboardX import SummaryWriter

from lib import model, common

import torch
import torch.optim as optim
import torch.nn.functional as F


ENV_ID = "MinitaurBulletEnv-v0"
GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
TGT_NET_SYNC = 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    save_path = os.path.join("saves", "ddpg-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_ID)

    net = model.ModelDDPG(env.observation_space.shape[0], env.action_space.shape[0])
    if args.cuda:
        net.cuda()
    print(net)
    tgt_net = ptan.agent.TargetNet(net)

    writer = SummaryWriter(comment="-ddpg_" + args.name)
    agent = model.AgentDDPG(net, cuda=args.cuda)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                frame_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    
                    mean_reward = tracker.reward(rewards[0], frame_idx)
                    if mean_reward is not None and (best_reward is None or best_reward < mean_reward):
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, mean_reward))
                            name = "best_%+.3f_%d.dat" % (mean_reward, frame_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(net.state_dict(), fname)
                        best_reward = mean_reward

                if len(buffer) < REPLAY_INITIAL:
                    continue

                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, dones_mask, last_states_v = \
                    common.unpack_batch_ddqn(batch, cuda=args.cuda)

                # train critic
                optimizer.zero_grad()
                q_v = net.critic(states_v, actions_v)
                q_last_v = tgt_net.target_model(last_states_v)[1]
                q_last_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()
                net.n_actor.zero_grad()
                optimizer.step()
                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)

                # train actor
                optimizer.zero_grad()
                cur_actions_v = net.actor(states_v)
                actor_loss_v = -net.critic(states_v, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                net.n_critic.zero_grad()
                optimizer.step()
                tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                if frame_idx % TGT_NET_SYNC == 0:
                    tgt_net.sync()
    pass
