from nltk.translate import bleu


def calc_bleu(cand_seq, ref_seq):
    return bleu([ref_seq], cand_seq, emulate_multibleu=True)


