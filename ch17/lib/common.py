import gym
import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils

DEFAULT_SEED = 20

NUM_ENVS = 16
GAMMA = 0.99
REWARD_STEPS = 5
ENTROPY_BETA = 0.01
VALUE_LOSS_COEF = 0.5
BATCH_SIZE = REWARD_STEPS * 16
CLIP_GRAD = 0.5

FRAMES_COUNT = 2
IMG_SHAPE = (FRAMES_COUNT, 84, 84)


def make_env(test=False, clip=True):
    if test:
        args = {'reward_clipping': False,
                'episodic_life': False}
    else:
        args = {'reward_clipping': clip}
    return ptan.common.wrappers.wrap_dqn(gym.make("BreakoutNoFrameskip-v4"),
                                         stack_frames=FRAMES_COUNT,
                                         **args)


def set_seed(seed, envs=None, cuda=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    if envs:
        for idx, env in enumerate(envs):
            env.seed(seed + idx)


class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        self.policy = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        fc_out = self.fc(conv_out)
        return self.policy(fc_out), self.value(fc_out)



def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


def iterate_batches(envs, net, device="cpu"):
    n_actions = envs[0].action_space.n
    act_selector = ptan.actions.ProbabilityActionSelector()
    obs = [e.reset() for e in envs]
    batch_dones = [[False] for _ in range(NUM_ENVS)]
    total_reward = [0.0] * NUM_ENVS
    total_steps = [0] * NUM_ENVS
    mb_obs = np.zeros((NUM_ENVS, REWARD_STEPS) + IMG_SHAPE, dtype=np.uint8)
    mb_rewards = np.zeros((NUM_ENVS, REWARD_STEPS), dtype=np.float32)
    mb_values = np.zeros((NUM_ENVS, REWARD_STEPS), dtype=np.float32)
    mb_actions = np.zeros((NUM_ENVS, REWARD_STEPS), dtype=np.int32)
    mb_probs = np.zeros((NUM_ENVS, REWARD_STEPS, n_actions), dtype=np.float32)

    while True:
        batch_dones = [[dones[-1]] for dones in batch_dones]
        done_rewards = []
        done_steps = []
        for n in range(REWARD_STEPS):
            obs_v = ptan.agent.default_states_preprocessor(obs).to(device)
            mb_obs[:, n] = obs_v.data.cpu().numpy()
            logits_v, values_v = net(obs_v)
            probs_v = F.softmax(logits_v, dim=1)
            probs = probs_v.data.cpu().numpy()
            actions = act_selector(probs)
            mb_probs[:, n] = probs
            mb_actions[:, n] = actions
            mb_values[:, n] = values_v.squeeze().data.cpu().numpy()
            for e_idx, e in enumerate(envs):
                o, r, done, _ = e.step(actions[e_idx])
                total_reward[e_idx] += r
                total_steps[e_idx] += 1
                if done:
                    o = e.reset()
                    done_rewards.append(total_reward[e_idx])
                    done_steps.append(total_steps[e_idx])
                    total_reward[e_idx] = 0.0
                    total_steps[e_idx] = 0
                obs[e_idx] = o
                mb_rewards[e_idx, n] = r
                batch_dones[e_idx].append(done)
        # obtain values for the last observation
        obs_v = ptan.agent.default_states_preprocessor(obs).to(device)
        _, values_v = net(obs_v)
        values_last = values_v.squeeze().data.cpu().numpy()

        for e_idx, (rewards, dones, value) in enumerate(zip(mb_rewards, batch_dones, values_last)):
            rewards = rewards.tolist()
            if not dones[-1]:
                rewards = discount_with_dones(rewards + [value], dones[1:] + [False], GAMMA)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones[1:], GAMMA)
            mb_rewards[e_idx] = rewards

        out_mb_obs = mb_obs.reshape((-1,) + IMG_SHAPE)
        out_mb_rewards = mb_rewards.flatten()
        out_mb_actions = mb_actions.flatten()
        out_mb_values = mb_values.flatten()
        out_mb_probs = mb_probs.flatten()
        yield out_mb_obs, out_mb_rewards, out_mb_actions, out_mb_values, out_mb_probs, \
              np.array(done_rewards), np.array(done_steps)


def train_a2c(net, mb_obs, mb_rewards, mb_actions, mb_values, optimizer, tb_tracker, step_idx, device="cpu"):
    optimizer.zero_grad()
    mb_adv = mb_rewards - mb_values
    adv_v = torch.FloatTensor(mb_adv).to(device)
    obs_v = torch.FloatTensor(mb_obs).to(device)
    rewards_v = torch.FloatTensor(mb_rewards).to(device)
    actions_t = torch.LongTensor(mb_actions).to(device)
    logits_v, values_v = net(obs_v)
    log_prob_v = F.log_softmax(logits_v, dim=1)
    log_prob_actions_v = adv_v * log_prob_v[range(len(mb_actions)), actions_t]

    loss_policy_v = -log_prob_actions_v.mean()
    loss_value_v = F.mse_loss(values_v.squeeze(-1), rewards_v)

    prob_v = F.softmax(logits_v, dim=1)
    entropy_loss_v = (prob_v * log_prob_v).sum(dim=1).mean()
    loss_v = ENTROPY_BETA * entropy_loss_v + VALUE_LOSS_COEF * loss_value_v + loss_policy_v
    loss_v.backward()
    nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
    optimizer.step()

    tb_tracker.track("advantage", mb_adv, step_idx)
    tb_tracker.track("values", values_v, step_idx)
    tb_tracker.track("batch_rewards", rewards_v, step_idx)
    tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
    tb_tracker.track("loss_policy", loss_policy_v, step_idx)
    tb_tracker.track("loss_value", loss_value_v, step_idx)
    tb_tracker.track("loss_total", loss_v, step_idx)
    return obs_v


def test_model(env, net, rounds=3, device="cpu"):
    total_reward = 0.0
    total_steps = 0
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True)

    for _ in range(rounds):
        obs = env.reset()
        while True:
            action = agent([obs])[0][0]
            obs, r, done, _ = env.step(action)
            total_reward += r
            total_steps += 1
            if done:
                break
    return total_reward / rounds, total_steps / rounds
