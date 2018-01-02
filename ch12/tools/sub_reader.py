#!/usr/bin/env python
import os
import sys
sys.path.append(os.getcwd())

import argparse

import xml.etree.ElementTree as ET

from libbots import subtitles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File to parse")
    args = parser.parse_args()

    tree = ET.parse(args.file)
    dialogues = subtitles.parse_dialogues(tree, dialog_seconds=3)
    print("Got %d dialogues:" % len(dialogues))
    for idx, dial in enumerate(dialogues):
        print()
        print("Dialogue %d with %d phrases:" % (idx, len(dial)))
        for phrase in dial:
            print(phrase)
    pass
