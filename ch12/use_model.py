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
    parser.add_argument("-m", "--model", required=True, help="Model name to load")
    parser.add_argument("-s", "--string", help="String to process, otherwise will loop")
    args = parser.parse_args()

    emb_dict, emb = data.load_embeddings(os.path.dirname(args.model))
    log.info("Embeddings loaded, shape=%s", emb.shape)

    embeddings = nn.Embedding(num_embeddings=emb.shape[0], embedding_dim=emb.shape[1])
    embeddings.weight.data.copy_(torch.from_numpy(emb))
    embeddings.weight.requires_grad = False

    net = model.PhraseModel(emb_size=emb.shape[1], dict_size=emb.shape[0], hid_size=model.HIDDEN_STATE_SIZE)
    net.load_state_dict(torch.load(args.model))

    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}

    if args.string:
        process_string(args.string, emb_dict, rev_emb_dict, embeddings, net)
    else:
        log.info("Enter phrase to process, empty string to exit")
        while True:
            s = input(">>> ")
            if not s:
                break
            process_string(s, emb_dict, rev_emb_dict, embeddings, net)
    pass
