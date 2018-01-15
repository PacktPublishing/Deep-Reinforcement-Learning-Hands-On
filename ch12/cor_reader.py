#!/usr/bin/env python3
import argparse
import collections

from libbots import cornell, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--genre", default='', help="Genre to show dialogs from")
    parser.add_argument("--show-genres", action='store_true', default=False, help="Display genres stats")
    parser.add_argument("--show-dials", action='store_true', default=False, help="Display dialogs")
    parser.add_argument("--show-train", action='store_true', default=False, help="Display training pairs")
    parser.add_argument("--show-dict-freq", action='store_true', default=False, help="Display dictionary frequency")
    args = parser.parse_args()

    if args.show_genres:
        genre_counts = collections.Counter()
        genres = cornell.read_genres(cornell.DATA_DIR)
        for movie, g_list in genres.items():
            for g in g_list:
                genre_counts[g] += 1
        print("Genres:")
        for g, count in genre_counts.most_common():
            print("%s: %d" % (g, count))

    if args.show_dials:
        dials = cornell.load_dialogues(genre_filter=args.genre)
        for d_idx, dial in enumerate(dials):
            print("Dialog %d with %d phrases:" % (d_idx, len(dial)))
            for p in dial:
                print(" ".join(p))
            print()

    if args.show_train or args.show_dict_freq:
        phrase_pairs, emb_dict = data.load_data(genre_filter=args.genre)

    if args.show_train:
        rev_emb_dict = {idx: word for word, idx in emb_dict.items()}
        train_data = data.encode_phrase_pairs(phrase_pairs, emb_dict)
        train_data = data.group_train_data(train_data)
        unk_token = emb_dict[data.UNKNOWN_TOKEN]

        print("Training pairs (%d total)" % len(train_data))
        train_data.sort(key=lambda p: len(p[1]), reverse=True)
        for idx, (p1, p2_group) in enumerate(train_data):
            w1 = data.decode_words(p1, rev_emb_dict)
            w2_group = [data.decode_words(p2, rev_emb_dict) for p2 in p2_group]
            print("%d:" % idx, " ".join(w1))
            for w2 in w2_group:
                print("%s:" % (" " * len(str(idx))), " ".join(w2))

    if args.show_dict_freq:
        words_stat = collections.Counter()
        for p1, p2 in phrase_pairs:
            words_stat.update(p1)
        print("Frequency stats for %d tokens in the dict" % len(emb_dict))
        for token, count in words_stat.most_common():
            print("%s: %d" % (token, count))
    pass
