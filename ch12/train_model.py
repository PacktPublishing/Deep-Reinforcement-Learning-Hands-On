#!/usr/bin/env python3
import argparse
import logging

from libbots import subtitles, data, model

DATA_FILE = "data/OpenSubtitles/en/Action/2005/365_100029_136606_sin_city.xml.gz"
HIDDEN_STATE_SIZE = 32

log = logging.getLogger("train")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action='store_true', default=False, help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    dialogues = subtitles.read_file(DATA_FILE, dialog_seconds=5)
    log.info("Loaded %d dialogues with %d phrases", len(dialogues), sum(map(len, dialogues)))
    emb_dict, emb = data.read_embeddings()
    train_data = data.dialogues_to_train(dialogues, emb_dict)
    log.info("Training data converted, got %d samples", len(train_data))

    net = model.PhraseModel(emb_size=emb.shape[1], dict_size=emb.shape[0], hid_size=HIDDEN_STATE_SIZE)
    if args.cuda:
        net.cuda()
    log.info("Model: %s", net)
    pass
