import collections
import os
import sys
import logging
import numpy as np
import pickle

from . import cornell, subtitles

EMBEDDINGS_FILE = "data/glove.6B.100d.txt"
UNKNOWN_TOKEN = '#UNK'
BEGIN_TOKEN = "#BEG"
END_TOKEN = "#END"
MAX_TOKENS = 10

EMB_DICT_NAME = "emb_dict.dat"
EMB_NAME = "emb.npy"

EMB_EXTRA = {
    UNKNOWN_TOKEN: 0.0,
    BEGIN_TOKEN: 1.0,
    END_TOKEN: -1.0
}

log = logging.getLogger("data")


def read_embeddings(word_set=None, file_name=EMBEDDINGS_FILE):
    """
    Read embeddings from text file: http://nlp.stanford.edu/data/glove.6B.zip
    :param word_set: set used to filter embeddings' dictionary
    :param file_name:
    :return: tuple with (dict word->id mapping, numpy matrix with embeddings)
    """
    log.info("Reading embeddings from %s", file_name)
    weights = []
    words = {}
    with open(file_name, "rt", encoding='utf-8') as fd:
        idx = 0
        for l in fd:
            v = l.split(' ')
            word, vec = v[0], list(map(float, v[1:]))
            if word_set is not None and word not in word_set:
                continue
            words[word] = idx
            # extra dim for our tokens
            vec.append(0.0)
            weights.append(vec)
            idx += 1
    for token, val in sorted(EMB_EXTRA.items()):
        words[token] = len(words)
        vec = [0.0]*(len(weights[0])-1)
        vec.append(val)
        weights.append(vec)
    emb = np.array(weights, dtype=np.float32)
    log.info("Embeddings loaded, shape=%s", emb.shape)
    return words, emb


def save_embeddings(dir_name, emb_dict, emb):
    with open(os.path.join(dir_name, EMB_DICT_NAME), "wb") as fd:
        pickle.dump(emb_dict, fd)
    np.save(os.path.join(dir_name, EMB_NAME), emb)


def load_embeddings(dir_name):
    with open(os.path.join(dir_name, EMB_DICT_NAME), "rb") as fd:
        emb_dict = pickle.load(fd)
    emb = np.load(os.path.join(dir_name, EMB_NAME))
    return emb_dict, emb


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


def encode_phrase_pairs(phrase_pairs, emb_dict):
    """
    Convert list of phrase pairs to training data
    :param phrase_pairs: list of (phrase, phrase)
    :param emb_dict: embeddings dictionary (word -> id)
    :return: list of tuples ([input_id_seq], [output_id_seq])
    """
    result = []
    for p1, p2 in phrase_pairs:
        p = encode_words(p1.words, emb_dict), encode_words(p2.words, emb_dict)
        result.append(p)
    return result


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


Phrase = collections.namedtuple("Phrase", field_names=('words', 'time_start', 'time_stop'))


def load_data(args, max_tokens=MAX_TOKENS):
    if args.cornell is not None:
        dialogues = cornell.load_dialogues(genre_filter=args.cornell)
    else:
        if args.data.endswith(".xml.gz"):
            dialogues = subtitles.read_file(args.data)
        elif len(args.data) == 0:
            dialogues = subtitles.read_dir(subtitles.DATA_DIR)
        else:
            data_path = os.path.join(subtitles.DATA_DIR, args.data)
            dialogues = subtitles.read_dir(data_path)
    if not dialogues:
        log.error("No dialogues found, exit!")
        sys.exit()
    log.info("Loaded %d dialogues with %d phrases, generating training pairs",
             len(dialogues), sum(map(len, dialogues)))
    phrase_pairs = subtitles.dialogues_to_pairs(dialogues, max_tokens=max_tokens)
    phrase_pairs_dict = subtitles.phrase_pairs_dict(phrase_pairs)
    return phrase_pairs, phrase_pairs_dict


def decode_words(indices, rev_emb_dict):
    return [rev_emb_dict.get(idx, UNKNOWN_TOKEN) for idx in indices]
