import collections
import os
import sys
import logging
import itertools
import pickle

from . import cornell

UNKNOWN_TOKEN = '#UNK'
BEGIN_TOKEN = "#BEG"
END_TOKEN = "#END"
MAX_TOKENS = 20
MIN_TOKEN_FEQ = 10
SHUFFLE_SEED = 5871

EMB_DICT_NAME = "emb_dict.dat"
EMB_NAME = "emb.npy"

log = logging.getLogger("data")


def save_emb_dict(dir_name, emb_dict):
    with open(os.path.join(dir_name, EMB_DICT_NAME), "wb") as fd:
        pickle.dump(emb_dict, fd)


def load_emb_dict(dir_name):
    with open(os.path.join(dir_name, EMB_DICT_NAME), "rb") as fd:
        return pickle.load(fd)


def encode_words(words, emb_dict):
    """
    Convert list of words into list of embeddings indices, adding our tokens
    :param words: list of strings
    :param emb_dict: embeddings dictionary
    :return: list of IDs
    """
    res = [emb_dict[BEGIN_TOKEN]]
    unk_idx = emb_dict[UNKNOWN_TOKEN]
    for w in words:
        idx = emb_dict.get(w.lower(), unk_idx)
        res.append(idx)
    res.append(emb_dict[END_TOKEN])
    return res


def encode_phrase_pairs(phrase_pairs, emb_dict, filter_unknows=True):
    """
    Convert list of phrase pairs to training data
    :param phrase_pairs: list of (phrase, phrase)
    :param emb_dict: embeddings dictionary (word -> id)
    :return: list of tuples ([input_id_seq], [output_id_seq])
    """
    unk_token = emb_dict[UNKNOWN_TOKEN]
    result = []
    for p1, p2 in phrase_pairs:
        p = encode_words(p1, emb_dict), encode_words(p2, emb_dict)
        if unk_token in p[0] or unk_token in p[1]:
            continue
        result.append(p)
    return result


def group_train_data(training_data):
    """
    Group training pairs by first phrase
    :param training_data: list of (seq1, seq2) pairs
    :return: list of (seq1, [seq*]) pairs
    """
    groups = collections.defaultdict(list)
    for p1, p2 in training_data:
        l = groups[tuple(p1)]
        l.append(p2)
    return list(groups.items())


def iterate_batches(data, batch_size):
    assert isinstance(data, list)
    assert isinstance(batch_size, int)

    ofs = 0
    while True:
        batch = data[ofs*batch_size:(ofs+1)*batch_size]
        if len(batch) <= 1:
            break
        yield batch
        ofs += 1


def load_data(genre_filter, max_tokens=MAX_TOKENS, min_token_freq=MIN_TOKEN_FEQ):
    dialogues = cornell.load_dialogues(genre_filter=genre_filter)
    if not dialogues:
        log.error("No dialogues found, exit!")
        sys.exit()
    log.info("Loaded %d dialogues with %d phrases, generating training pairs",
             len(dialogues), sum(map(len, dialogues)))
    phrase_pairs = dialogues_to_pairs(dialogues, max_tokens=max_tokens)
    log.info("Counting freq of words...")
    word_counts = collections.Counter()
    for dial in dialogues:
        for p in dial:
            word_counts.update(p)
    freq_set = set(map(lambda p: p[0], filter(lambda p: p[1] >= min_token_freq, word_counts.items())))
    log.info("Data has %d uniq words, %d of them occur more than %d",
             len(word_counts), len(freq_set), min_token_freq)
    phrase_dict = phrase_pairs_dict(phrase_pairs, freq_set)
    return phrase_pairs, phrase_dict


def phrase_pairs_dict(phrase_pairs, freq_set):
    """
    Return the dict of words in the dialogues mapped to their IDs
    :param phrase_pairs: list of (phrase, phrase) pairs
    :return: dict
    """
    res = {UNKNOWN_TOKEN: 0, BEGIN_TOKEN: 1, END_TOKEN: 2}
    next_id = 3
    for p1, p2 in phrase_pairs:
        for w in map(str.lower, itertools.chain(p1, p2)):
            if w not in res and w in freq_set:
                res[w] = next_id
                next_id += 1
    return res


def dialogues_to_pairs(dialogues, max_tokens=None):
    """
    Convert dialogues to training pairs of phrases
    :param dialogues:
    :param max_tokens: limit of tokens in both question and reply
    :return: list of (phrase, phrase) pairs
    """
    result = []
    for dial in dialogues:
        prev_phrase = None
        for phrase in dial:
            if prev_phrase is not None:
                if max_tokens is None or (len(prev_phrase) <= max_tokens and len(phrase) <= max_tokens):
                    result.append((prev_phrase, phrase))
            prev_phrase = phrase
    return result


def decode_words(indices, rev_emb_dict):
    return [rev_emb_dict.get(idx, UNKNOWN_TOKEN) for idx in indices]


def trim_tokens_seq(tokens, end_token):
    res = []
    for t in tokens:
        res.append(t)
        if t == end_token:
            break
    return res


def split_train_test(data, train_ratio=0.95):
    count = int(len(data) * train_ratio)
    return data[:count], data[count:]
