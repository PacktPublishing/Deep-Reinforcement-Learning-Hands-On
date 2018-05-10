#!/usr/bin/env python3
import os
import random
import argparse
import logging
import numpy as np
from tensorboardX import SummaryWriter

from libbots import data, model, utils

import torch
import torch.optim as optim
import torch.nn.functional as F

SAVES_DIR = "saves"

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHES = 100

log = logging.getLogger("train")

TEACHER_PROB = 0.5


def run_test(test_data, net, end_token, device="cpu"):
    bleu_sum = 0.0
    bleu_count = 0
    for p1, p2 in test_data:
        input_seq = model.pack_input(p1, net.emb, device)
        enc = net.encode(input_seq)
        _, tokens = net.decode_chain_argmax(enc, input_seq.data[0:1],
                                            seq_len=data.MAX_TOKENS,
                                            stop_at_token=end_token)
        bleu_sum += utils.calc_bleu(tokens, p2[1:])
        bleu_count += 1
    return bleu_sum / bleu_count


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Category to use for training. "
                                                      "Empty string to train on full dataset")
    parser.add_argument("--cuda", action='store_true', default=False,
                        help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    saves_path = os.path.join(SAVES_DIR, args.name)
    os.makedirs(saves_path, exist_ok=True)

    phrase_pairs, emb_dict = data.load_data(genre_filter=args.data)
    log.info("Obtained %d phrase pairs with %d uniq words",
             len(phrase_pairs), len(emb_dict))
    data.save_emb_dict(saves_path, emb_dict)
    end_token = emb_dict[data.END_TOKEN]
    train_data = data.encode_phrase_pairs(phrase_pairs, emb_dict)
    rand = np.random.RandomState(data.SHUFFLE_SEED)
    rand.shuffle(train_data)
    log.info("Training data converted, got %d samples", len(train_data))
    train_data, test_data = data.split_train_test(train_data)
    log.info("Train set has %d phrases, test %d", len(train_data), len(test_data))

    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict),
                            hid_size=model.HIDDEN_STATE_SIZE).to(device)
    log.info("Model: %s", net)

    writer = SummaryWriter(comment="-" + args.name)

    optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    best_bleu = None
    for epoch in range(MAX_EPOCHES):
        losses = []
        bleu_sum = 0.0
        bleu_count = 0
        for batch in data.iterate_batches(train_data, BATCH_SIZE):
            optimiser.zero_grad()
            input_seq, out_seq_list, _, out_idx = model.pack_batch(batch, net.emb, device)
            enc = net.encode(input_seq)

            net_results = []
            net_targets = []
            for idx, out_seq in enumerate(out_seq_list):
                ref_indices = out_idx[idx][1:]
                enc_item = net.get_encoded_item(enc, idx)
                if random.random() < TEACHER_PROB:
                    r = net.decode_teacher(enc_item, out_seq)
                    bleu_sum += model.seq_bleu(r, ref_indices)
                else:
                    r, seq = net.decode_chain_argmax(enc_item, out_seq.data[0:1],
                                                     len(ref_indices))
                    bleu_sum += utils.calc_bleu(seq, ref_indices)
                net_results.append(r)
                net_targets.extend(ref_indices)
                bleu_count += 1
            results_v = torch.cat(net_results)
            targets_v = torch.LongTensor(net_targets).to(device)
            loss_v = F.cross_entropy(results_v, targets_v)
            loss_v.backward()
            optimiser.step()

            losses.append(loss_v.item())
        bleu = bleu_sum / bleu_count
        bleu_test = run_test(test_data, net, end_token, device)
        log.info("Epoch %d: mean loss %.3f, mean BLEU %.3f, test BLEU %.3f",
                 epoch, np.mean(losses), bleu, bleu_test)
        writer.add_scalar("loss", np.mean(losses), epoch)
        writer.add_scalar("bleu", bleu, epoch)
        writer.add_scalar("bleu_test", bleu_test, epoch)
        if best_bleu is None or best_bleu < bleu_test:
            if best_bleu is not None:
                out_name = os.path.join(saves_path, "pre_bleu_%.3f_%02d.dat" %
                                        (bleu_test, epoch))
                torch.save(net.state_dict(), out_name)
                log.info("Best BLEU updated %.3f", bleu_test)
            best_bleu = bleu_test

        if epoch % 10 == 0:
            out_name = os.path.join(saves_path, "epoch_%03d_%.3f_%.3f.dat" %
                                    (epoch, bleu, bleu_test))
            torch.save(net.state_dict(), out_name)

    writer.close()
