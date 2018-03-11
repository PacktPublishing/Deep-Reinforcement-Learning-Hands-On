"""
Monte-Carlo Tree Search
"""
import math as m
import numpy as np

from lib import game, model


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

    def search(self, state_int, player, net, tau, cuda=False):
        """
        Performs a search until the leaf node
        :param state_int: root node to start the search
        :param player: player to move
        :param net: network with model used for leaf expansion
        :return: action probabilities for the root node
        """
        actions = []
        states = []
        cur_state = state_int
        cur_player = player
        player_won = None
        draw = False

        while not self.is_leaf(cur_state):
            states.append(cur_state)

            counts = self.visit_count[cur_state]
            total_sqrt = m.sqrt(sum(counts))
            probs = self.probs[cur_state]
            values_avg = self.value_avg[cur_state]

            # choose action to take
            score = [value + self.c_puct * prob * total_sqrt / (1 + count)
                     for value, prob, count in zip(values_avg, probs, counts)]
            action = np.argmax(score)
            actions.append(action)
            cur_state, won = game.move(cur_state, action, cur_player)
            if won:
                player_won = cur_player == player
                break
            cur_player = 1-cur_player
            # check for the draw
            if len(game.possible_moves(cur_state)) == 0:
                draw = True
                break

        # TODO: maybe we don't need to expand the terminal game nodes?

        # we have a leaf node in cur_state, expand it
        state_list = game.decode_binary(cur_state)
        batch_v = model.state_lists_to_batch([state_list], [cur_player], cuda)
        probs_v, values_v = net(batch_v)

        # save the node
        self.visit_count[cur_state] = [0] * game.GAME_COLS
        self.value[cur_state] = [0.0] * game.GAME_COLS
        self.value_avg[cur_state] = [0.0] * game.GAME_COLS
        self.probs[cur_state] = probs_v[0].data.cpu().numpy().tolist()

        # back up the value
        value = values_v[0].data.cpu().numpy()[0]
        for state_int, action in zip(states, actions):
            self.visit_count[state_int][action] += 1
            self.value[state_int][action] += value
            self.value_avg[state_int][action] = self.value[state_int][action] / self.visit_count[state_int][action]

        # calculate the action probabilities for the root node
        root_counts = [count ** (1.0/tau) for count in self.visit_count[state_int]]
        root_total = sum(root_counts)
        if root_total > 0:
            root_probs = [count / root_total for count in root_counts]
        else:
            root_probs = [1 / game.GAME_COLS] * game.GAME_COLS
        return root_probs

    def is_leaf(self, state_int):
        return state_int not in self.probs


pass
