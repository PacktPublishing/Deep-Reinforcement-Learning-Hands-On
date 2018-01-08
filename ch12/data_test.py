#!/usr/bin/env python3
import os
import re
import argparse
import logging

from libbots import data, model

import torch
import torch.nn as nn

log = logging.getLogger("use")


def process_string(s, emb_dict, rev_emb_dict, embeddings, net, max_out=20):
    words = re.split(r'\s+', s)
    words = list(map(str.lower, words))
    tokens = data.encode_words(words, emb_dict)
    log.info("Words: %s, tokens: %s", words, tokens)
    input_seq = model.pack_input(tokens, embeddings)
    enc = net.encode(input_seq)

    # decoding
    cur_token = data.BEGIN_TOKEN
    unk_idx = emb_dict[data.UNKNOWN_TOKEN]
    end_idx = emb_dict[data.END_TOKEN]
    result = []

    while True:
        cur_idx = emb_dict.get(cur_token, unk_idx)
        cur_emb = embeddings(torch.LongTensor([cur_idx]))
        out_logits, new_enc = net.decode_one(enc, cur_emb)
        out_token_v = torch.max(out_logits, dim=1)[1]
        out_token = out_token_v.data.cpu().numpy()[0]
        out_word = rev_emb_dict.get(out_token, data.UNKNOWN_TOKEN)
        result.append(out_word)

        if out_token == end_idx:
            break
        if out_token == cur_token:
            break
        if len(result) > max_out:
            break

        cur_token = out_token
        enc = new_enc

    print(result)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cornell", required=True, help="Use Cornell Movie Dialogues database could be "
                                                         "a category or empty string to load full data")
    parser.add_argument("-m", "--model", required=True, help="Model name to load")
    parser.add_argument("-s", "--string", help="String to process, otherwise will loop")
    args = parser.parse_args()

    phrase_pairs, emb_dict = data.load_data(args)
    data.extend_emb_dict(emb_dict)
    log.info("Obtained %d phrase pairs with %d uniq words", len(phrase_pairs), len(emb_dict))
    train_data = data.encode_phrase_pairs(phrase_pairs, emb_dict)
    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}

    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict), hid_size=model.HIDDEN_STATE_SIZE)
    net.load_state_dict(torch.load(args.model))

    for seq_1, seq_2 in train_data:
        pass
    pass
