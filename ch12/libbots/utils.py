import collections


def iterate_n_grams(seq, n):
    if n == 1:
        yield from seq
        return
    q = collections.deque(maxlen=n)
    for s in seq:
        q.append(s)
        if len(q) == n:
            yield tuple(q)


def calc_bleu(cand_seq, ref_seq, max_n_grams=4):
    """
    Calculate BLEU score
    :param cand_seq:
    :param ref_seq:
    :return:
    """
    score = 0.0

    for n in range(1, max_n_grams+1):
        cand_ngrams = list(iterate_n_grams(cand_seq, n=n))
        cand_counts = collections.Counter(cand_ngrams)
        ref_counts = collections.Counter(iterate_n_grams(ref_seq, n=n))
        for item, count in cand_counts.items():
            if item in ref_counts:
                score += min(count, ref_counts[item]) / len(cand_ngrams)
    score /= max_n_grams
    return score


