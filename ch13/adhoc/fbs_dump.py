#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd())
sys.path.append("..")
import argparse

from lib.ksy import fbs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fbs", required=True, help="Name of the FBS file to read")
    parser.add_argument("-o", "--output", required=True,
                        help="Name of data file to dump RFP stream")
    args = parser.parse_args()

    fbs_file = fbs.Fbs.from_file(args.fbs)
    with open(args.output, 'wb') as fd:
        for b in fbs_file.blocks:
            fd.write(b.data)
    pass
