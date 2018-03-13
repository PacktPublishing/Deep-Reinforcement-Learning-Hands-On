#!/usr/bin/env python3
import os
import time
import ptan
import random
import argparse
import collections
import numpy as np

from lib import game, model, mcts

from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


MCTS_SEARCHES = 3
MCTS_BATCH_SIZE = 30
REPLAY_BUFFER = 10000
LEARNING_RATE = 0.1
BATCH_SIZE = 256
TRAIN_ROUNDS = 10
MIN_REPLAY_TO_TRAIN = REPLAY_BUFFER

BEST_NET_WIN_RATIO = 0.6

EVALUATE_EVERY_STEP = 50
EVALUATION_ROUNDS = 100
STEPS_BEFORE_TAU_0 = 10


def play_game(mcts_store, replay_buffer, net1, net2, cuda=False):
    """
    Play one single game, memorizing transitions into the replay buffer
    :param replay_buffer: queue with (state, probs, values), if None, nothing is stored
    :param net1: player1
    :param net2: player2
    :return: value for the game in respect to player1 (+1 if p1 won, -1 if lost, 0 if draw)
    """
    assert isinstance(replay_buffer, (collections.deque, type(None)))
    state = game.INITIAL_STATE
    nets = [net1, net2]
    cur_player = np.random.choice(2)
    step = 0
    result = None
    tau = 1
    while result is None:
        mcts_store.search_batch(MCTS_SEARCHES, MCTS_BATCH_SIZE, state, cur_player, nets[cur_player], cuda=cuda)
        probs, values = mcts_store.get_policy_value(state, tau=tau)
        if replay_buffer is not None:
            replay_buffer.append((state, cur_player, probs, values))
        action = np.random.choice(game.GAME_COLS, p=probs)
        if action not in game.possible_moves(state):
            print("Impossible action selected")
        state, won = game.move(state, action, cur_player)
        if won:
            result = 1.0 if cur_player == 0 else -1
        cur_player = 1-cur_player
        # check the draw case
        if len(game.possible_moves(state)) == 0:
            result = 0.0
        step += 1
        if step >= STEPS_BEFORE_TAU_0:
            tau = 0
    return result, step


def evaluate(net1, net2, rounds, cuda=False):
    n1_win, n2_win = 0, 0
    for r_idx in range(rounds):
        if r_idx % 2 == 0:
            r = model.play_game(net1, net2, cuda)
        else:
            r = -model.play_game(net2, net1, cuda)
        if r < -0.5:
            n2_win += 1
        elif r > 0.5:
            n1_win += 1
    return n1_win / (n1_win + n2_win)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    args = parser.parse_args()

    saves_path = os.path.join("saves", args.name)
    os.makedirs(saves_path, exist_ok=True)
    writer = SummaryWriter(comment="-" + args.name)

    net = model.Net(input_shape=model.OBS_SHAPE, actions_n=game.GAME_COLS)
    if args.cuda:
        net.cuda()
    best_net = ptan.agent.TargetNet(net)
    print(net)

    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER)
    mcts_store = mcts.MCTS()
    step_idx = 0
    best_idx = 0

    with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
        while True:
            t = time.time()
            prev_nodes = len(mcts_store)
            game_res, game_steps = play_game(mcts_store, replay_buffer, best_net.target_model,
                                             best_net.target_model, cuda=args.cuda)
            game_nodes = len(mcts_store) - prev_nodes
            dt = time.time() - t
            speed_steps = game_steps / dt
            speed_nodes = game_nodes / dt
            tb_tracker.track("speed_steps", speed_steps, step_idx)
            tb_tracker.track("speed_nodes", speed_nodes, step_idx)
            print("Game %d, steps %3d, leaves %4d, steps/s %5.2f, leaves/s %6.2f, best_idx %d, replay %d" % (
                step_idx, game_steps, game_nodes, speed_steps, speed_nodes, best_idx, len(replay_buffer)))
            step_idx += 1

            if len(replay_buffer) < MIN_REPLAY_TO_TRAIN:
                continue


            # train
            sum_loss = 0.0
            sum_value_loss = 0.0
            sum_policy_loss = 0.0

            for _ in range(TRAIN_ROUNDS):
                batch = random.sample(replay_buffer, BATCH_SIZE)
                batch_states, batch_who_moves, batch_probs, batch_values = zip(*batch)
                batch_states_lists = [game.decode_binary(state) for state in batch_states]
                states_v = model.state_lists_to_batch(batch_states_lists, batch_who_moves, args.cuda)

                optimizer.zero_grad()
                probs_v = Variable(torch.FloatTensor(batch_probs))
                values_v = Variable(torch.FloatTensor(batch_values))
                if args.cuda:
                    probs_v = probs_v.cuda()
                    values_v = values_v.cuda()
                # obtain expected value for the state
                state_value_v = (probs_v * values_v).sum(dim=1).detach()

                out_logits_v, out_values_v = net(states_v)

                loss_value_v = F.mse_loss(out_values_v, state_value_v)
                loss_policy_v = -F.log_softmax(out_logits_v) * probs_v
                loss_policy_v = loss_policy_v.sum(dim=1).mean()

                loss_v = loss_policy_v + loss_value_v
                loss_v.backward()
                optimizer.step()
                sum_loss += loss_v.data.cpu().numpy()[0]
                sum_value_loss += loss_value_v.data.cpu().numpy()[0]
                sum_policy_loss += loss_policy_v.data.cpu().numpy()[0]

            tb_tracker.track("loss_total", sum_loss / TRAIN_ROUNDS, step_idx)
            tb_tracker.track("loss_value", sum_value_loss / TRAIN_ROUNDS, step_idx)
            tb_tracker.track("loss_policy", sum_policy_loss / TRAIN_ROUNDS, step_idx)

            # evaluate net
            if step_idx % EVALUATE_EVERY_STEP == 0:
                win_ratio = evaluate(net, best_net.target_model, rounds=EVALUATION_ROUNDS, cuda=args.cuda)
                print("Net evaluated, win ratio = %.2f" % win_ratio)
                writer.add_scalar("eval_win_ratio", win_ratio, step_idx)
                if win_ratio > BEST_NET_WIN_RATIO:
                    print("Net is better than cur best, sync")
                    best_net.sync()
                    best_idx += 1
                    file_name = os.path.join(saves_path, "best_%03d_%05d.dat" % (best_idx, step_idx))
                    torch.save(net.state_dict(), file_name)
#                    replay_buffer.clear()
                    mcts_store.clear()
