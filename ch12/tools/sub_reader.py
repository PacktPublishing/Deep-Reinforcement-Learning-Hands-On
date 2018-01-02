#!/usr/bin/env python
import os
import sys
import argparse
import logging
import collections

sys.path.append(os.getcwd())

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
        count_not_found = collections.Counter()

        for dial in dialogues:
            for phrase in dial:
                for w in phrase.words:
                    if w.lower() not in emb_dict:
                        count_not_found[w.lower()] += 1

        log.info("Not found %d tokens in embeddings, 10 most common:", len(count_not_found))
        for token, count in count_not_found.most_common(10):
            log.info("%s: %d", token, count)
