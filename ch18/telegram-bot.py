#!/usr/bin/env python3
# This module requires python-telegram-bot
import os
import sys
import glob
import json
import time
import datetime
import random
import logging
import numpy as np
import configparser
import argparse

from lib import game, model, mcts

MCTS_SEARCHES = 20
MCTS_BATCH_SIZE = 4

try:
    import telegram.ext
    from telegram.error import TimedOut
except ImportError:
    print("You need python-telegram-bot package installed to start the bot")
    sys.exit()

import torch

# Configuration file with the following contents
# [telegram]
# api=API_KEY
CONFIG_DEFAULT = "~/.config/rl_ch18_bot.ini"

log = logging.getLogger("telegram")


class Session:
    BOT_PLAYER = game.PLAYER_BLACK
    USER_PLAYER = game.PLAYER_WHITE

    def __init__(self, model_file, player_moves_first, player_id):
        self.model_file = model_file
        self.model = model.Net(input_shape=model.OBS_SHAPE, actions_n=game.GAME_COLS)
        self.model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        self.state = game.INITIAL_STATE
        self.value = None
        self.player_moves_first = player_moves_first
        self.player_id = player_id
        self.moves = []
        self.mcts_store = mcts.MCTS()

    def move_player(self, col):
        self.moves.append(col)
        self.state, won = game.move(self.state, col, self.USER_PLAYER)
        return won

    def move_bot(self):
        self.mcts_store.search_batch(MCTS_SEARCHES, MCTS_BATCH_SIZE, self.state, self.BOT_PLAYER, self.model)
        probs, values = self.mcts_store.get_policy_value(self.state, tau=0)
        action = np.random.choice(game.GAME_COLS, p=probs)
        self.value = values[action]
        self.moves.append(action)
        self.state, won = game.move(self.state, action, self.BOT_PLAYER)
        return won

    def is_valid_move(self, move_col):
        return move_col in game.possible_moves(self.state)

    def is_draw(self):
        return len(game.possible_moves(self.state)) == 0

    def render(self):
        l = game.render(self.state)
        l = "\n".join(l)
        l = l.replace("0", 'O').replace("1", "X")
        board = "0123456\n-------\n" + l + "\n-------\n0123456"
        extra = ""
        if self.value is not None:
            extra = "Position evaluation: %.2f\n" % float(self.value)
        return extra + "<pre>%s</pre>" % board


class PlayerBot:
    def __init__(self, models_dir, log_file):
        self.sessions = {}
        self.models_dir = models_dir
        self.models = self._read_models(models_dir)
        self.log_file = log_file
        self.leaderboard = {}
        self._read_leaderboard(log_file)

    def _read_models(self, models_dir):
        result = {}
        for idx, name in enumerate(sorted(glob.glob(os.path.join(models_dir, "*.dat")))):
            result[idx] = name
        return result

    def _read_leaderboard(self, log_file):
        if not os.path.exists(log_file):
            return 
        with open(log_file, 'rt', encoding='utf-8') as fd:
            for l in fd:
                data = json.loads(l)
                bot_name = os.path.basename(data['model_file'])
                user_name = data['player_id'].split(':')[0]
                bot_score = data['bot_score']
                self._update_leaderboard(bot_score, bot_name, user_name)

    def _update_leaderboard(self, bot_score, bot_name, user_name):
        if bot_score > 0.5:
            game.update_counts(self.leaderboard, bot_name, (1, 0, 0))
            game.update_counts(self.leaderboard, user_name, (0, 1, 0))
        elif bot_score < -0.5:
            game.update_counts(self.leaderboard, bot_name, (0, 1, 0))
            game.update_counts(self.leaderboard, user_name, (1, 0, 0))
        else:
            game.update_counts(self.leaderboard, bot_name, (0, 0, 1))
            game.update_counts(self.leaderboard, user_name, (0, 0, 1))

    def _save_log(self, session, bot_score):
        self._update_leaderboard(bot_score, os.path.basename(session.model_file),
                                 session.player_id.split(':')[0])
        data = {
            "ts": time.time(),
            "time": datetime.datetime.utcnow().isoformat(),
            "bot_score": bot_score,
            "model_file": session.model_file,
            "player_id": session.player_id,
            "player_moves_first": session.player_moves_first,
            "moves": session.moves,
            "state": session.state
        }
        with open(self.log_file, "a+t", encoding='utf-8') as f:
            f.write(json.dumps(data, sort_keys=True) + '\n')

    def command_help(self, bot, update):
        bot.send_message(chat_id=update.message.chat_id, parse_mode="HTML", disable_web_page_preview=True,
                         text="""
This a <a href="https://en.wikipedia.org/wiki/Connect_Four">4-in-a-row</a> game bot trained with AlphaGo Zero method for the <a href="https://www.packtpub.com/big-data-and-business-intelligence/practical-deep-reinforcement-learning">Practical Deep Reinforcement Learning</a> book. 

<b>Welcome!</b>

This bot understands the following commands:
<b>/list</b> to list available pre-trained models (the higher the ID, the stronger the play)
<b>/play MODEL_ID</b> to start the new game against the specified model
<b>/top</b> show leaderboard

During the game, your moves are numbers of columns to drop the disk.
""")


    def command_list(self, bot, update):
        if len(self.models) == 0:
            reply = ["There are no models currently available, sorry!"]
        else:
            reply = ["The list of available models with their IDs"]
            for idx, name in sorted(self.models.items()):
                reply.append("<b>%d</b>: %s" % (idx, os.path.basename(name)))

        bot.send_message(chat_id=update.message.chat_id, text="\n".join(reply), parse_mode="HTML")

    def command_play(self, bot, update, args):
        chat_id = update.message.chat_id
        player_id = "%s:%s" % (update.message.from_user.username, update.message.from_user.id)
        try:
            model_id = int(args[0])
        except ValueError:
            bot.send_message(chat_id=chat_id, text="Wrong argumants! Use '/play <MODEL_ID>, to start the game")
            return

        if model_id not in self.models:
            bot.send_message(chat_id=chat_id, text="There is no such model, use /list command to get list of IDs")
            return

        if chat_id in self.sessions:
            bot.send_message(chat_id=chat_id, text="You already have the game in progress, it will be discarded")
            del self.sessions[chat_id]

        player_moves = random.choice([False, True])
        session = Session(self.models[model_id], player_moves, player_id)
        self.sessions[chat_id] = session
        if player_moves:
            bot.send_message(chat_id=chat_id, text="Your move is first (you're playing with O), please give the column to put your checker - single number from 0 to 6")
        else:
            bot.send_message(chat_id=chat_id, text="The first move is mine (I'm playing with X), moving...")
            session.move_bot()
        bot.send_message(chat_id=chat_id, text=session.render(), parse_mode="HTML")

    def text(self, bot, update):
        chat_id = update.message.chat_id

        if chat_id not in self.sessions:
            bot.send_message(chat_id=chat_id, text="You have no game in progress. Start it with <b>/play MODEL_ID</b> "
                                                   "(or use <b>/help</b> to see the list of commands)",
                             parse_mode='HTML')
            return
        session = self.sessions[chat_id]

        try:
            move_col = int(update.message.text)
        except ValueError:
            bot.send_message(chat_id=chat_id, text="I don't understand. In play mode you can give a number "
                                                   "from 0 to 6 to specify your move.")
            return

        if move_col < 0 or move_col > game.GAME_COLS:
            bot.send_message(chat_id=chat_id, text="Wrong column specified! It must be in range 0-6")
            return

        if not session.is_valid_move(move_col):
            bot.send_message(chat_id=chat_id, text="Move %d is invalid!" % move_col)
            return

        won = session.move_player(move_col)
        if won:
            bot.send_message(chat_id=chat_id, text="You won! Congratulations!")
            self._save_log(session, bot_score=-1)
            del self.sessions[chat_id]
            return

        won = session.move_bot()
        bot.send_message(chat_id=chat_id, text=session.render(), parse_mode="HTML")

        if won:
            bot.send_message(chat_id=chat_id, text="I won! Wheeee!")
            self._save_log(session, bot_score=1)
            del self.sessions[chat_id]
        # checking for a draw
        if session.is_draw():
            bot.send_message(chat_id=chat_id, text="Draw position. That's unlikely, but possible. 1:1, see ya!")
            self._save_log(session, bot_score=0)
            del self.sessions[chat_id]

    def error(self, bot, update, error):
        try:
            raise error
        except TimedOut:
            log.info("Timed out error")

    def command_top(self, bot, update):
        res = ["Leader board"]
        items = list(self.leaderboard.items())
        items.sort(reverse=True, key=lambda p: p[1][0])
        for user, (wins, losses, draws) in items:
            res.append("%20s: won=%d, lost=%d, draw=%d" % (user[:20], wins, losses, draws))
        l = "\n".join(res)
        bot.send_message(chat_id=update.message.chat_id, text="<pre>" + l + "</pre>", parse_mode="HTML")

    def command_refresh(self, bot, update):
        self.models = self._read_models(self.models_dir)
        bot.send_message(chat_id=update.message.chat_id, text="Models reloaded, %d files have found" % len(self.models))


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_DEFAULT,
                        help="Configuration file for the bot, default=" + CONFIG_DEFAULT)
    parser.add_argument("-m", "--models", required=True, help="Directory name with models to serve")
    parser.add_argument("-l", "--log", required=True, help="Log name to keep the games and leaderboard")
    prog_args = parser.parse_args()

    conf = configparser.ConfigParser()
    if not conf.read(os.path.expanduser(prog_args.config)):
        log.error("Configuration file %s not found", prog_args.config)
        sys.exit()

    player_bot = PlayerBot(prog_args.models, prog_args.log)

    updater = telegram.ext.Updater(conf['telegram']['api'])
    updater.dispatcher.add_handler(telegram.ext.CommandHandler('help', player_bot.command_help))
    updater.dispatcher.add_handler(telegram.ext.CommandHandler('list', player_bot.command_list))
    updater.dispatcher.add_handler(telegram.ext.CommandHandler('top', player_bot.command_top))
    updater.dispatcher.add_handler(telegram.ext.CommandHandler('play', player_bot.command_play, pass_args=True))
    updater.dispatcher.add_handler(telegram.ext.CommandHandler('refresh', player_bot.command_refresh))
    updater.dispatcher.add_handler(telegram.ext.MessageHandler(telegram.ext.Filters.text, player_bot.text))
    updater.dispatcher.add_error_handler(player_bot.error)

    log.info("Bot initialized, started serving")
    updater.start_polling()
    updater.idle()

    pass
