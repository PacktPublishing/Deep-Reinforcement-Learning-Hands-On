#!/usr/bin/env python3
import argparse
import logging

from libbots import data, model, utils

import torch

log = logging.getLogger("data_test")

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="Category to use for training. Empty string to train on full dataset")
    parser.add_argument("-m", "--model", required=True, help="Model name to load")
    args = parser.parse_args()

    phrase_pairs, emb_dict = data.load_data(args.data)
    log.info("Obtained %d phrase pairs with %d uniq words", len(phrase_pairs), len(emb_dict))
    train_data = data.encode_phrase_pairs(phrase_pairs, emb_dict)
    train_data = data.group_train_data(train_data)
    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}

    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict), hid_size=model.HIDDEN_STATE_SIZE)
    net.load_state_dict(torch.load(args.model))

    end_token = emb_dict[data.END_TOKEN]

    seq_count = 0
    sum_bleu = 0.0

    for seq_1, targets in train_data:
        input_seq = model.pack_input(seq_1, net.emb)
        enc = net.encode(input_seq)
        _, tokens = net.decode_chain_argmax(enc, input_seq.data[0:1],
                                            seq_len=data.MAX_TOKENS, stop_at_token=end_token)
        references = [seq[1:] for seq in targets]
        bleu = utils.calc_bleu_many(tokens, references)
        sum_bleu += bleu
        seq_count += 1

    log.info("Processed %d phrases, mean BLEU = %.4f", seq_count, sum_bleu / seq_count)
