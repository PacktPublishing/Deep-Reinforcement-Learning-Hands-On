#!/usr/bin/env python3
# This module requires python-telegram-bot
import os
import sys
import logging
import configparser
import argparse
from nltk.tokenize import TweetTokenizer

try:
    import telegram.ext
except ImportError:
    print("You need python-telegram-bot package installed to start the bot")
    sys.exit()

from libbots import data, model

import torch

# Configuration file with the following contents
# [telegram]
# api=API_KEY
CONFIG_DEFAULT = "~/.config/rl_ch12_bot.ini"

log = logging.getLogger("telegram")



if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_DEFAULT,
                        help="Configuration file for the bot, default=" + CONFIG_DEFAULT)
    parser.add_argument("-m", "--model", required=True, help="Model to load")
    args = parser.parse_args()

    conf = configparser.ConfigParser()
    if not conf.read(os.path.expanduser(args.config)):
        log.error("Configuration file %s not found", args.config)
        sys.exit()

    emb_dict = data.load_emb_dict(os.path.dirname(args.model))
    log.info("Loaded embedded dict with %d entries", len(emb_dict))
    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}
    end_token = emb_dict[data.END_TOKEN]

    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict), hid_size=model.HIDDEN_STATE_SIZE)
    net.load_state_dict(torch.load(args.model))
    tokeniser = TweetTokenizer(preserve_case=False)

    def bot_func(bot, update, args):
        text = " ".join(args)
        words = tokeniser.tokenize(text)
        seq_1 = data.encode_words(words, emb_dict)
        input_seq = model.pack_input(seq_1, net.emb)
        enc = net.encode(input_seq)
        _, tokens = net.decode_chain_argmax(net.emb, enc, input_seq.data[0:1],
                                            seq_len=data.MAX_TOKENS, stop_at_token=end_token)
        if tokens[-1] == end_token:
            tokens = tokens[:-1]
        reply = data.decode_words(tokens, rev_emb_dict)
        if reply:
            bot.send_message(chat_id=update.message.chat_id, text=" ".join(reply))

    updater = telegram.ext.Updater(conf['telegram']['api'])
    updater.dispatcher.add_handler(telegram.ext.CommandHandler('bot', bot_func, pass_args=True))

    log.info("Bot initialized, started serving")
    updater.start_polling()
    updater.idle()

    pass
