#!/usr/bin/env python
import os
import sys
sys.path.append(os.getcwd())

import argparse
import logging

from libbots import subtitles, data

log = logging.getLogger("sub_reader")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogues", action='store_true', default=False, help="Dump dialogues")
    parser.add_argument("--embeddings", action='store_true', default=False, help="Check embeddings stats")
    parser.add_argument("file", help="File to parse")
    args = parser.parse_args()

    dialogues = subtitles.read_file(args.file, dialog_seconds=3)
    log.info("Got %d dialogues", len(dialogues))

    if args.dialogues:
        for idx, dial in enumerate(dialogues):
            print()
            print("Dialogue %d with %d phrases:" % (idx, len(dial)))
            for phrase in dial:
                print(phrase)

    if args.embeddings:
        emb_dict, emb = data.read_embeddings()



    pass
