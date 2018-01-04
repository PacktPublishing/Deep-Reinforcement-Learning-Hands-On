#!/usr/bin/env python3
import os
import random
import argparse
import logging
import numpy as np
from tensorboardX import SummaryWriter

from libbots import subtitles, data, model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


SAVES_DIR = "saves"
DATA_FILE = "data/OpenSubtitles/en/Action/2005/365_100029_136606_sin_city.xml.gz"
HIDDEN_STATE_SIZE = 128
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
MAX_EPOCHES = 100

log = logging.getLogger("train")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action='store_true', default=False, help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    saves_path = os.path.join(SAVES_DIR, args.name)
    os.makedirs(saves_path, exist_ok=True)

    dialogues = subtitles.read_file(DATA_FILE, dialog_seconds=5)
    dial_dict = subtitles.dialogues_dict(dialogues)
    log.info("Loaded %d dialogues with %d phrases and %d uniq words",
             len(dialogues), sum(map(len, dialogues)), len(dial_dict))
    emb_dict, emb = data.read_embeddings(dial_dict)
    train_data = data.dialogues_to_train(dialogues, emb_dict)
    log.info("Training data converted, got %d samples", len(train_data))

    # initialize embedding lookup table
    embeddings = nn.Embedding(num_embeddings=emb.shape[0], embedding_dim=emb.shape[1])
    embeddings.weight.data.copy_(torch.from_numpy(emb))
    embeddings.weight.requires_grad = False
    if args.cuda:
        embeddings.cuda()

    net = model.PhraseModel(emb_size=emb.shape[1], dict_size=emb.shape[0], hid_size=HIDDEN_STATE_SIZE)
    if args.cuda:
        net.cuda()
    log.info("Model: %s", net)

    writer = SummaryWriter(comment="-" + args.name)
    optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    for epoch in range(MAX_EPOCHES):
        random.shuffle(train_data)
        losses = []
        bleu_sum = 0.0
        bleu_count = 0
        for batch in data.iterate_batches(train_data, BATCH_SIZE):
            input_seq, out_seq_list, out_idx = model.pack_batch(batch, embeddings, cuda=args.cuda)
            enc = net.encode(input_seq)

            net_results = []
            net_targets = []
            for idx, out_seq in enumerate(out_seq_list):
                r = net.decode_teacher(net.get_encoded_item(enc, idx), out_seq)
                net_results.append(r)
                net_targets.extend(out_idx[idx][1:])
                bleu_sum += model.seq_bleu(r, out_idx[idx][1:])
                bleu_count += 1
            results_v = torch.cat(net_results)
            targets_v = Variable(torch.LongTensor(net_targets))
            if args.cuda:
                targets_v = targets_v.cuda()
            loss_v = F.cross_entropy(results_v, targets_v)
            loss_v.backward()
            optimiser.step()

            losses.append(loss_v.data.cpu().numpy()[0])
        log.info("Epoch %d: mean loss %.3f, mean BLEU %.3f", epoch, np.mean(losses), bleu_sum / bleu_count)
        writer.add_scalar("loss", np.mean(losses), epoch)
        writer.add_scalar("bleu", bleu_sum / bleu_count, epoch)
        torch.save(net.state_dict(), os.path.join(saves_path, "pre_%02d.dat" % epoch))
    writer.close()
