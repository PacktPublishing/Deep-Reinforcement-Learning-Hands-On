import os
import logging
import numpy as np
import pickle

EMBEDDINGS_FILE = "data/glove.6B.100d.txt"
UNKNOWN_TOKEN = '#UNK'

log = logging.getLogger("data")


def read_embeddings(file_name=EMBEDDINGS_FILE):
    """
    Read embeddings from text file: http://nlp.stanford.edu/data/glove.6B.zip
    :param file_name:
    :return: tuple with (dict word->id mapping, numpy matrix with embeddings)
    """
    emb_file_name = file_name + ".emb.npy"
    dic_file_name = file_name + ".dic"
    if os.path.exists(emb_file_name) and os.path.exists(dic_file_name):
        log.info("Loading cached embeddings from %s and %s", emb_file_name, dic_file_name)
        with open(dic_file_name, 'rb') as fd:
            words = pickle.load(fd)
        emb = np.load(emb_file_name)
    else:
        log.info("Reading embeddings from %s", file_name)
        weights = []
        words = {UNKNOWN_TOKEN: 0}
        with open(file_name, "rt", encoding='utf-8') as fd:
            for idx, l in enumerate(fd):
                v = l.split(' ')
                word, vec = v[0], list(map(float, v[1:]))
                words[word] = idx+1
                weights.append(vec)
        weights.insert(0, [0.0]*len(weights[0]))
        emb = np.array(weights, dtype=np.float32)
        with open(dic_file_name, "wb") as fd:
            pickle.dump(words, fd)
        np.save(emb_file_name, emb)
    log.info("Embeddings loaded, shape=%s", emb.shape)
    return words, emb
