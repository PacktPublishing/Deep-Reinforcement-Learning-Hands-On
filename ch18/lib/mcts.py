"""
Monte-Carlo Tree Search
"""
import math as m
import numpy as np

from lib import game, model

import torch.nn.functional as F


class MCTS:
    """
    Class keeps statistics for every state encountered during the search
    """
    def __init__(self, c_puct=1.0):
        self.c_puct = c_puct
        # count of visits, state_int -> [N(s, a)]
        self.visit_count = {}
        # total value of the state's action, state_int -> [W(s, a)]
        self.value = {}
        # average value of actions, state_int -> [Q(s, a)]
        self.value_avg = {}
        # prior probability of actions, state_int -> [P(s,a)]
        self.probs = {}

    def search(self, state_int, player, net, cuda=False):
        """
        Performs one round of MCTS search, including the leaf expansion and backfill
        :param state_int: root node to start the search
        :param player: player to move
        :param net: network with model used for leaf expansion
        """
        actions = []
        states = []
        cur_state = state_int
        cur_player = player
        value = None

        while not self.is_leaf(cur_state):
            states.append(cur_state)

            counts = self.visit_count[cur_state]
            total_sqrt = m.sqrt(sum(counts))
            probs = self.probs[cur_state]
            values_avg = self.value_avg[cur_state]

            # choose action to take, in the root node add the Dirichlet noise to the probs
            if cur_state == state_int:
                noises = np.random.dirichlet([0.03] * game.GAME_COLS)
                probs = [0.75 * prob + 0.25 * noise for prob, noise in zip(probs, noises)]
            score = [value + self.c_puct * prob * total_sqrt / (1 + count)
                     for value, prob, count in zip(values_avg, probs, counts)]
            invalid_actions = set(range(game.GAME_COLS)) - set(game.possible_moves(cur_state))
            for invalid in invalid_actions:
                score[invalid] = -np.inf
            action = int(np.argmax(score))
            actions.append(action)
            cur_state, won = game.move(cur_state, action, cur_player)
            if won:
                value = 1.0 if cur_player == player else -1
                break
            cur_player = 1-cur_player
            # check for the draw
            if len(game.possible_moves(cur_state)) == 0:
                value = 0.0
                break

        # haven't reached the end of match, expand leaf node
        if value is None:
            # we have a leaf node in cur_state, expand it
            state_list = game.decode_binary(cur_state)
            batch_v = model.state_lists_to_batch([state_list], [cur_player], cuda)
            logits_v, values_v = net(batch_v)
            probs_v = F.softmax(logits_v)
            value = values_v[0].data.cpu().numpy()[0]

            # save the node
            self.visit_count[cur_state] = [0] * game.GAME_COLS
            self.value[cur_state] = [0.0] * game.GAME_COLS
            self.value_avg[cur_state] = [0.0] * game.GAME_COLS
            self.probs[cur_state] = probs_v[0].data.cpu().numpy().tolist()

        # back up the obtained value
        for state_int, action in zip(states, actions):
            self.visit_count[state_int][action] += 1
            self.value[state_int][action] += value
            self.value_avg[state_int][action] = self.value[state_int][action] / self.visit_count[state_int][action]
        #
        # # calculate the action probabilities for the root node
        # root_counts = [count ** (1.0/tau) for count in self.visit_count[state_int]]
        # root_total = sum(root_counts)
        # if root_total > 0:
        #     root_probs = [count / root_total for count in root_counts]
        # else:
        #     root_probs = [1 / game.GAME_COLS] * game.GAME_COLS
        # return root_probs

    def is_leaf(self, state_int):
        return state_int not in self.probs

    def search_batch(self, count, state_int, player, net):
        """
        Perform several MCTS searches. 
        TODO: optimize this using one single NN pass
        """
        for _ in range(count):
            self.search(state_int, player, net)

    def get_policy_value(self, state_int, tau=1):
        """
        Extract policy and action-values by the state
        :param state_int: state of the board
        :return: (probs, values)
        """
        counts = self.visit_count[state_int]
        if tau == 0:
            probs = [0.0] * game.GAME_COLS
            probs[np.argmax(counts)] = 1.0
        else:
            counts = [count ** (1.0 / tau) for count in counts]
            total = sum(counts)
            probs = [count / total for count in counts]
        values = self.value_avg[state_int]
        return probs, values

pass
