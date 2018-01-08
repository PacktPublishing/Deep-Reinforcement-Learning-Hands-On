#!/usr/bin/env python3
import os
import random
import argparse
import logging
import numpy as np
from tensorboardX import SummaryWriter

from libbots import data, model, utils

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import ptan

DEFAULT_FILE = "data/OpenSubtitles/en/Crime/1994/60_101020_138057_pulp_fiction.xml.gz"
SAVES_DIR = "saves"

BATCH_SIZE = 64
LEARNING_RATE = 1e-5
MAX_EPOCHES = 10000

log = logging.getLogger("train")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cornell", help="Use Cornell Movie Dialogues database could be "
                                          "a category or empty string to load full data")
    parser.add_argument("--data", default=DEFAULT_FILE, help="Could be file name to load or category dir")
    parser.add_argument("--cuda", action='store_true', default=False, help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-l", "--load", required=True, help="Load model and continue in RL mode")
    parser.add_argument("--samples", type=int, default=1, help="Count of samples in prob mode")
    args = parser.parse_args()

    saves_path = os.path.join(SAVES_DIR, args.name)
    os.makedirs(saves_path, exist_ok=True)

    phrase_pairs, emb_dict = data.load_data(args)
    log.info("Obtained %d phrase pairs with %d uniq words", len(phrase_pairs), len(emb_dict))
    data.save_emb_dict(saves_path, emb_dict)
    data.extend_emb_dict(emb_dict)
    end_token = emb_dict[data.END_TOKEN]
    train_data = data.encode_phrase_pairs(phrase_pairs, emb_dict)
    log.info("Training data converted, got %d samples", len(train_data))

    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}

    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict), hid_size=model.HIDDEN_STATE_SIZE)
    if args.cuda:
        net.cuda()
    log.info("Model: %s", net)

    writer = SummaryWriter(comment="-" + args.name)
    net.load_state_dict(torch.load(args.load))
    log.info("Model loaded from %s, continue training in RL mode...", args.load)

    with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
        optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
        batch_idx = 0
        best_bleu = None
        for epoch in range(MAX_EPOCHES):
            random.shuffle(train_data)
            dial_shown = False

            total_samples = 0
            skipped_samples = 0
            bleus_argmax = []
            bleus_sample = []

            for batch in data.iterate_batches(train_data, BATCH_SIZE):
                batch_idx += 1
                optimiser.zero_grad()
                input_seq, out_seq_list, inp_idx, out_idx = model.pack_batch(batch, net.emb, cuda=args.cuda)
                enc = net.encode(input_seq)

                net_policies = []
                net_actions = []
                net_advantages = []

                for idx, out_seq in enumerate(out_seq_list):
                    total_samples += 1
                    ref_indices = out_idx[idx][1:]
                    item_enc = net.get_encoded_item(enc, idx)
                    r_argmax, actions = net.decode_chain_argmax(net.emb, item_enc, out_seq.data[0],
                                                                data.MAX_TOKENS*2, stop_at_token=end_token)
                    # if perfect match, skip the sample
                    if ref_indices == actions:
                        skipped_samples += 1
                        continue

                    argmax_bleu = utils.calc_bleu(actions, ref_indices)
                    bleus_argmax.append(argmax_bleu)

                    if not dial_shown:
                        log.info("Input: %s", " ".join(data.decode_words(inp_idx[idx], rev_emb_dict)))
                        log.info("Refer: %s", " ".join(data.decode_words(ref_indices, rev_emb_dict)))
                        log.info("Argmax: %s, bleu=%.4f", " ".join(data.decode_words(actions, rev_emb_dict)),
                                 argmax_bleu)

                    for _ in range(args.samples):
                        r_sample, actions = net.decode_chain_sampling(net.emb, item_enc, out_seq.data[0],
                                                                      data.MAX_TOKENS*2, stop_at_token=end_token)
                        sample_bleu = utils.calc_bleu(actions, ref_indices)

                        if not dial_shown:
                            log.info("Sample: %s, bleu=%.4f", " ".join(data.decode_words(actions, rev_emb_dict)),
                                     sample_bleu)

                        net_policies.append(r_sample)
                        net_actions.extend(actions)
                        net_advantages.extend([sample_bleu - argmax_bleu] * len(actions))
                        bleus_sample.append(sample_bleu)
                    dial_shown = True

                if not net_policies:
                    continue

                policies_v = torch.cat(net_policies)
                actions_t = torch.LongTensor(net_actions)
                adv_v = Variable(torch.FloatTensor(net_advantages))
                if args.cuda:
                    actions_t = actions_t.cuda()
                    adv_v = adv_v.cuda()

                log_prob_v = F.log_softmax(policies_v)
                log_prob_actions_v = adv_v * log_prob_v[range(len(net_actions)), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                loss_v = loss_policy_v
                loss_v.backward()
                optimiser.step()

                tb_tracker.track("advantage", adv_v, batch_idx)
                tb_tracker.track("loss_policy", loss_policy_v, batch_idx)
                tb_tracker.track("loss_total", loss_v, batch_idx)

            bleu = np.mean(bleus_argmax + [1.0] * skipped_samples)
            writer.add_scalar("bleu", bleu, batch_idx)
            writer.add_scalar("bleu_argmax", np.mean(bleus_argmax), batch_idx)
            writer.add_scalar("bleu_sample", np.mean(bleus_sample), batch_idx)
            writer.add_scalar("skipped_samples", skipped_samples / total_samples, batch_idx)
            writer.add_scalar("epoch", batch_idx, epoch)
            log.info("Epoch %d", epoch)
            if best_bleu is None or best_bleu < bleu:
                best_bleu = bleu
                log.info("Best bleu updated: %.4f", bleu)
                torch.save(net.state_dict(), os.path.join(saves_path, "bleu_%.3f_%02d.dat" % (bleu, epoch)))

    writer.close()
