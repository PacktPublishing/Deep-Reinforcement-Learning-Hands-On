#!/usr/bin/env python3
import os
import argparse
import logging

from libbots import data, model, utils

import torch

log = logging.getLogger("use")


def words_to_words(words, emb_dict, rev_emb_dict, net, use_sampling=False):
    tokens = data.encode_words(words, emb_dict)
    input_seq = model.pack_input(tokens, net.emb)
    enc = net.encode(input_seq)
    end_token = emb_dict[data.END_TOKEN]
    if use_sampling:
        _, out_tokens = net.decode_chain_sampling(enc, input_seq.data[0:1], seq_len=data.MAX_TOKENS,
                                                  stop_at_token=end_token)
    else:
        _, out_tokens = net.decode_chain_argmax(enc, input_seq.data[0:1], seq_len=data.MAX_TOKENS,
                                                stop_at_token=end_token)
    if out_tokens[-1] == end_token:
        out_tokens = out_tokens[:-1]
    out_words = data.decode_words(out_tokens, rev_emb_dict)
    return out_words


def process_string(s, emb_dict, rev_emb_dict, net, use_sampling=False):
    out_words = words_to_words(words, emb_dict, rev_emb_dict, net, use_sampling=use_sampling)
    print(" ".join(out_words))


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model name to load")
    parser.add_argument("-s", "--string", help="String to process, otherwise will loop")
    parser.add_argument("--sample", default=False, action="store_true", help="Enable sampling generation instead of argmax")
    parser.add_argument("--self", type=int, default=1, help="Enable self-loop mode with given amount of phrases.")
    args = parser.parse_args()

    emb_dict = data.load_emb_dict(os.path.dirname(args.model))
    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict), hid_size=model.HIDDEN_STATE_SIZE)
    net.load_state_dict(torch.load(args.model))

    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}

    while True:
        if args.string:
            input_string = args.string
        else:
            input_string = input(">>> ")
        if not input_string:
            break

        words = utils.tokenize(input_string)
        for _ in range(args.self):
            words = words_to_words(words, emb_dict, rev_emb_dict, net, use_sampling=args.sample)
            print(utils.untokenize(words))

        if args.string:
            break
    pass
