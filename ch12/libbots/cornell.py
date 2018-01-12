"""
Cornel Movies Dialogs Corpus
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
"""
import os
import logging

from . import utils

log = logging.getLogger("cornell")
DATA_DIR = "data/cornell"
SEPARATOR = "+++$+++"


def load_dialogues(data_dir=DATA_DIR, genre_filter=''):
    """
    Load dialogues from cornell data
    :return: list of list of list of words
    """
    movie_set = None
    if genre_filter:
        movie_set = read_movie_set(data_dir, genre_filter)
        log.info("Loaded %d movies with genre %s", len(movie_set), genre_filter)
    log.info("Read and tokenise phrases...")
    lines = read_phrases(data_dir, movies=movie_set)
    log.info("Loaded %d phrases", len(lines))
    dialogues = load_conversations(data_dir, lines, movie_set)
    return dialogues


def iterate_entries(data_dir, file_name):
    with open(os.path.join(data_dir, file_name), "rb") as fd:
        for l in fd:
            l = str(l, encoding='utf-8', errors='ignore')
            yield list(map(str.strip, l.split(SEPARATOR)))


def read_movie_set(data_dir, genre_filter):
    res = set()
    for parts in iterate_entries(data_dir, "movie_titles_metadata.txt"):
        m_id, m_genres = parts[0], parts[5]
        if m_genres.find(genre_filter) != -1:
            res.add(m_id)
    return res


def read_phrases(data_dir, movies=None):
    res = {}
    for parts in iterate_entries(data_dir, "movie_lines.txt"):
        l_id, m_id, l_str = parts[0], parts[2], parts[4]
        if movies and m_id not in movies:
            continue
        tokens = utils.tokenize(l_str)
        if tokens:
            res[l_id] = tokens
    return res


def load_conversations(data_dir, lines, movies=None):
    res = []
    for parts in iterate_entries(data_dir, "movie_conversations.txt"):
        m_id, dial_s = parts[2], parts[3]
        if movies and m_id not in movies:
            continue
        l_ids = dial_s.strip("[]").split(", ")
        l_ids = list(map(lambda s: s.strip("'"), l_ids))
        dial = [lines[l_id] for l_id in l_ids if l_id in lines]
        if dial:
            res.append(dial)
    return res


def read_genres(data_dir):
    res = {}
    for parts in iterate_entries(data_dir, "movie_titles_metadata.txt"):
        m_id, m_genres = parts[0], parts[5]
        l_genres = m_genres.strip("[]").split(", ")
        l_genres = list(map(lambda s: s.strip("'"), l_genres))
        res[m_id] = l_genres
    return res
