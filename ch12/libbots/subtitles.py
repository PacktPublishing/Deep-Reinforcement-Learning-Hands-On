import os
import gzip
import glob
import datetime
import xml.etree.ElementTree as ET

from . import data

DATA_DIR = "data/OpenSubtitles/en/"


def read_dir(dir_name):
    result = []
    for file_name in glob.glob(os.path.join(dir_name, "**/*.xml.gz"), recursive=True):
        result.extend(read_file(file_name))
    return result


def read_file(file_name, **kwargs):
    if file_name.endswith('.gz'):
        with gzip.open(file_name) as fd:
            tree = ET.parse(fd)
    else:
        tree = ET.parse(file_name)
    return parse_dialogues(tree, **kwargs)


def parse_dialogues(tree, dialog_seconds=5):
    """
    Convert XML tree into list of dialogues
    :param tree:
    :param dialog_seconds: delay between dialogues
    :return: list of lists of phrases
    """
    res = []
    cur_dialogue = []
    prev_phrase = None

    for phrase in iterate_phrases(tree.getroot()):
        if prev_phrase is None:
            prev_phrase = phrase
            continue
        cur_dialogue.extend(split_phrase(prev_phrase))
        delta = phrase.time_start - prev_phrase.time_stop
        if delta.total_seconds() > dialog_seconds:
            cur_dialogue.extend(split_phrase(phrase))
            if len(cur_dialogue) > 1:
                res.append(preprocess_dialogue(cur_dialogue))
                cur_dialogue = []
            phrase = None
        prev_phrase = phrase

    if prev_phrase is not None:
        cur_dialogue.extend(split_phrase(prev_phrase))
    res.append(preprocess_dialogue(cur_dialogue))
    return res


def iterate_phrases(elem_iter):
    time_start = None
    time_stop = None
    words = []
    for elem in elem_iter:
        for sub_elem in elem:
            if sub_elem.tag == "time":
                t_val = sub_elem.attrib['value']
                t_id = sub_elem.attrib['id']
                if t_id[-1] == 'S':
                    time_start = parse_time(t_val)
                elif t_id[-1] == 'E':
                    time_stop = parse_time(t_val)
                    words = remove_braced_words(words)
                    if words:
                        yield data.Phrase(words=words, time_start=time_start, time_stop=time_stop)
                    words = []
                else:
                    print("Unknown id: %s" % t_id)
            elif sub_elem.tag == "w":
                words.append(sub_elem.text)


def parse_time(time_str):
    h_str, m_str, sec_str = time_str.split(':')
    sec_str, msec_str = sec_str.split(',')
    if not msec_str:
        msec_str = '0'
    return datetime.timedelta(hours=int(h_str), minutes=int(m_str), seconds=int(sec_str), milliseconds=int(msec_str))


def remove_braced_words(words):
    """
    Drops words sublist within square or round brackets
    :param words: list of words
    :return: list of words
    """
    res = []
    in_brackets = False
    for w in words:
        if w in {'[', '('}:
            in_brackets = True
            continue
        if w in {']', ')'}:
            in_brackets = False
            continue
        if in_brackets:
            continue
        res.append(w)
    return res


def split_phrase(phrase):
    """
    Split phrase by dashes used to
    :param phrase:
    :return:
    """
    assert isinstance(phrase, data.Phrase)
    parts = []
    cur_part = []
    for w in phrase.words:
        if w == '-':
            if cur_part:
                parts.append(cur_part)
                cur_part = []
        else:
            cur_part.append(w)
    if cur_part:
        parts.append(cur_part)
    if len(parts) == 0:
        return []
    if len(parts) == 1:
        return [data.Phrase(words=parts[0], time_start=phrase.time_start, time_stop=phrase.time_stop)]
    delta = (phrase.time_stop - phrase.time_start) / len(parts)
    result = []
    for idx, part in enumerate(parts):
        result.append(data.Phrase(words=part, time_start=phrase.time_start + idx * delta,
                             time_stop=phrase.time_start + (idx+1)*delta))
    return result


def phrase_expand_abbrevs(phrase):
    """
    Expand abbreviations in-place
    """
    for idx, w in enumerate(phrase.words):
        if w.endswith("'") and idx < len(phrase.words)-1:
            lw = w.lower()
            lww = phrase.words[idx + 1].lower()

            if w.endswith("n'"):
                phrase.words[idx] = w[:-2]
                phrase.words[idx+1] = 'not'
            elif lww == 're':
                phrase.words[idx] = w[:-1]
                phrase.words[idx+1] = 'are'
            elif lww == 've':
                phrase.words[idx] = w[:-1]
                phrase.words[idx+1] = 'have'
            elif lww == 's':    # horrible thing to do
                phrase.words[idx] = w[:-1]
                phrase.words[idx + 1] = 'is'
            elif lww == 'll' or lww == 'il':
                phrase.words[idx] = w[:-1]
                phrase.words[idx + 1] = 'will'
            elif lw == "i'":
                if lww == "m":
                    phrase.words[idx] = 'I'
                    phrase.words[idx+1] = 'am'


def preprocess_dialogue(dialogue):
    """
    :param dialogue: list of phrases
    :return: list of phrases
    """
    for phrase in dialogue:
        phrase_expand_abbrevs(phrase)
    return dialogue


def phrase_pairs_dict(phrase_pairs):
    """
    Return set of words in the dialogues
    :param phrase_pairs: list of (phrase, phrase) pairs
    :return: set
    """
    res = set()
    for p1, p2 in phrase_pairs:
        res |= set(map(str.lower, p1.words)) | set(map(str.lower, p2.words))
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
                if max_tokens is None or (len(prev_phrase.words) <= max_tokens and len(phrase.words) <= max_tokens):
                    result.append((prev_phrase, phrase))
            prev_phrase = phrase
    return result
