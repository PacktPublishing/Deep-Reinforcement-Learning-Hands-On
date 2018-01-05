#!/usr/bin/env python3
import os
import sys
import argparse
import collections
sys.path.append(os.getcwd())

from libbots import cornell


if __name__ == "__main__":
    genre_counts = collections.Counter()
    genres = cornell.read_genres(cornell.DATA_DIR)
    for movie, g_list in genres.items():
        for g in g_list:
            genre_counts[g] += 1
    for g, count in genre_counts.most_common():
        print("%s: %d" % (g, count))
    pass
