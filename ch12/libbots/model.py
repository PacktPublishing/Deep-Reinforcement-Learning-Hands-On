import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable


class PhraseModel(nn.Module):
    def __init__(self, emb_size, dict_size, hid_size):
        super(PhraseModel, self).__init__()

        self.encoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size, num_layers=1, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hid_size, dict_size)
        )

    def encode(self, x):
        _, hid = self.encoder(x)
        return hid

    def get_encoded_item(self, encoded, index):
        # For RNN
        # return encoded[:, index:index+1]
        # For LSTM
        return encoded[0][:, index:index+1], encoded[1][:, index:index+1]

    def decode_teacher(self, hid, input_seq):
        # Method assumes batch of size=1
        out, _ = self.decoder(input_seq, hid)
        out = self.output(out.data)
        return out


def pack_batch(batch, embeddings, cuda=False):
    assert isinstance(batch, list)
    # Sort descending (CuDNN requirements)
    batch.sort(key=lambda s: len(s[0]), reverse=True)
    input_idx, output_idx = zip(*batch)
    # create padded matrix of inputs
    lens = list(map(len, input_idx))
    input_mat = np.zeros((len(batch), lens[0]), dtype=np.int64)
    for idx, x in enumerate(input_idx):
        input_mat[idx, :len(x)] = x
    input_v = Variable(torch.from_numpy(input_mat))
    if cuda:
        input_v = input_v.cuda()
    input_seq = rnn_utils.pack_padded_sequence(input_v, lens, batch_first=True)
    # lookup embeddings
    r = embeddings(input_seq.data)
    emb_input_seq = rnn_utils.PackedSequence(data=r, batch_sizes=input_seq.batch_sizes)

    # prepare output sequences, with end token stripped
    output_seq_list = []
    for out in output_idx:
        seq = out[:-1]
        out_v = Variable(torch.LongTensor([seq]))
        if cuda:
            out_v = out_v.cuda()
        emb_out_v = embeddings(out_v)
        out_seq = rnn_utils.pack_padded_sequence(emb_out_v, [len(seq)], batch_first=True)
        output_seq_list.append(out_seq)
    return emb_input_seq, output_seq_list, output_idx
