#!/usr/bin/env python3
import os
import random
import argparse
import logging
from tensorboardX import SummaryWriter

from libbots import data, model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import ptan

DEFAULT_FILE = "data/OpenSubtitles/en/Crime/1994/60_101020_138057_pulp_fiction.xml.gz"
SAVES_DIR = "saves"

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHES = 1000
MAX_TOKENS = 15

log = logging.getLogger("train")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cornell", help="Use Cornell Movie Dialogues database could be a category or empty string to load full data")
    parser.add_argument("--data", default=DEFAULT_FILE, help="Could be file name to load or category dir")
    parser.add_argument("--cuda", action='store_true', default=False, help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-l", "--load", required=True, help="Load model and continue in RL mode")
    args = parser.parse_args()

    saves_path = os.path.join(SAVES_DIR, args.name)
    os.makedirs(saves_path, exist_ok=True)

    phrase_pairs, phrase_pairs_dict = data.load_data(args, MAX_TOKENS)
    log.info("Obtained %d phrase pairs with %d uniq words", len(phrase_pairs), len(phrase_pairs_dict))
    emb_dict, emb = data.read_embeddings(phrase_pairs_dict)
    train_data = data.encode_phrase_pairs(phrase_pairs, emb_dict)
    log.info("Training data converted, got %d samples", len(train_data))

    data.save_embeddings(saves_path, emb_dict, emb)

    # initialize embedding lookup table
    embeddings = nn.Embedding(num_embeddings=emb.shape[0], embedding_dim=emb.shape[1])
    embeddings.weight.data.copy_(torch.from_numpy(emb))
    embeddings.weight.requires_grad = False
    if args.cuda:
        embeddings.cuda()

    net = model.PhraseModel(emb_size=emb.shape[1], dict_size=emb.shape[0], hid_size=model.HIDDEN_STATE_SIZE)
    if args.cuda:
        net.cuda()
    log.info("Model: %s", net)

    writer = SummaryWriter(comment="-" + args.name)
    net.load_state_dict(torch.load(args.load))
    log.info("Model loaded from %s, continue training in RL mode...", args.load)

    with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
        optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
        batch_idx = 0
        for epoch in range(MAX_EPOCHES):
            random.shuffle(train_data)
            epoch_bleu = 0.0
            epoch_bleu_count = 0

            for batch in data.iterate_batches(train_data, BATCH_SIZE):
                batch_idx += 1
                optimiser.zero_grad()
                input_seq, out_seq_list, out_idx = model.pack_batch(batch, embeddings, cuda=args.cuda)
                enc = net.encode(input_seq)

                net_policies = []
                net_actions = []
                net_advantages = []
                sum_argmax_bleu = 0.0
                sum_sample_bleu = 0.0

                for idx, out_seq in enumerate(out_seq_list):
                    ref_indices = out_idx[idx][1:]
                    item_enc = net.get_encoded_item(enc, idx)
                    r_argmax, _ = net.decode_chain_argmax(embeddings, item_enc, out_seq.data[0], len(ref_indices))
                    argmax_bleu = model.seq_bleu(r_argmax, ref_indices)
                    r_sample, actions = net.decode_chain_sampling(embeddings, item_enc, out_seq.data[0], len(ref_indices))
                    sample_bleu = model.seq_bleu(r_sample, ref_indices)

                    net_policies.append(r_sample)
                    net_actions.extend(actions)
                    net_advantages.extend([sample_bleu - argmax_bleu] * len(actions))
                    sum_argmax_bleu += argmax_bleu
                    sum_sample_bleu += sample_bleu

                policies_v = torch.cat(net_policies)
                actions_t = torch.LongTensor(net_actions)
                adv_v = Variable(torch.FloatTensor(net_advantages))
                if args.cuda:
                    actions_t = actions_t.cuda()
                    adv_v = adv_v.cuda()

                log_prob_v = F.log_softmax(policies_v)
                log_prob_actions_v = adv_v * log_prob_v[range(len(net_actions)), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                # prob_v = F.softmax(policies_v)
                # entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()
                # loss_v = entropy_loss_v + loss_policy_v
                loss_v = loss_policy_v
                loss_v.backward()
                optimiser.step()

                tb_tracker.track("bleu_argmax", sum_argmax_bleu / len(out_seq_list), batch_idx)
                tb_tracker.track("bleu_sample", sum_sample_bleu / len(out_seq_list), batch_idx)
                tb_tracker.track("advantage", adv_v, batch_idx)
                # tb_tracker.track("loss_entropy", entropy_loss_v, batch_idx)
                tb_tracker.track("loss_policy", loss_policy_v, batch_idx)
                tb_tracker.track("loss_total", loss_v, batch_idx)

                epoch_bleu += sum_sample_bleu / len(out_seq_list)
                epoch_bleu_count += 1
            log.info("Epoch %d: mean BLEU: %.3f", epoch, epoch_bleu / epoch_bleu_count)


    writer.close()
