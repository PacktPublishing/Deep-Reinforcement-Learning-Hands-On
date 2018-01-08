from nltk.translate import bleu_score


def calc_bleu(cand_seq, ref_seq):
    sf = bleu_score.SmoothingFunction()
    return bleu_score.sentence_bleu([ref_seq], cand_seq,
                                    smoothing_function=sf.method1,
                                    weights=(0.5, 0.5))


