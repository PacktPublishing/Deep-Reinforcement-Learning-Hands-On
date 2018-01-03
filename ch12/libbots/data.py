import os
import logging
import numpy as np
import pickle

EMBEDDINGS_FILE = "data/glove.6B.100d.txt"
UNKNOWN_TOKEN = '#UNK'
BEGIN_TOKEN = "#BEG"
END_TOKEN = "#END"

EMB_EXTRA = {
    UNKNOWN_TOKEN: 0.0,
    BEGIN_TOKEN: 1.0,
    END_TOKEN: -1.0
}

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
        words = {}
        with open(file_name, "rt", encoding='utf-8') as fd:
            for idx, l in enumerate(fd):
                v = l.split(' ')
                word, vec = v[0], list(map(float, v[1:]))
                words[word] = idx
                # extra dim for our tokens
                vec.append(0.0)
                weights.append(vec)
        for token, val in sorted(EMB_EXTRA.items()):
            words[token] = len(words)
            vec = [0.0]*(len(weights[0])-1)
            vec.append(val)
            weights.append(vec)
        emb = np.array(weights, dtype=np.float32)
        with open(dic_file_name, "wb") as fd:
            pickle.dump(words, fd)
        np.save(emb_file_name, emb)
    log.info("Embeddings loaded, shape=%s", emb.shape)
    return words, emb


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


def dialogues_to_train(dialogues, emb_dict):
    """
    Convert list of dialogues to training data
    :param dialogues: list of list of Phrase objects
    :param emb_dict: embeddings dictionary (word -> id)
    :return: list of tuples ([input_id_seq], [output_id_seq])
    """
    result = []
    for dial in dialogues:
        prev_phrase = None
        for phrase in dial:
            enc_phrase = encode_words(phrase.words, emb_dict)
            if prev_phrase is not None:
                result.append((prev_phrase, enc_phrase))
            prev_phrase = enc_phrase
    return result

