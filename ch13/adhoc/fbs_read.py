#!/usr/bin/env python3
import sys
sys.path.append("..")

import struct
import argparse
from universe.vncdriver.fbs_reader import FBSReader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fbs", required=True, help="Name of the FBS file to read")
    args = parser.parse_args()

    reader = FBSReader(args.fbs)
    for idx, (dat, ts) in enumerate(reader):
        msg = struct.unpack("!B", dat[0:1])[0]
        print(idx, len(dat), ts, msg)

    pass
