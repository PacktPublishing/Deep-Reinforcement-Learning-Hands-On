import logging
import pickle
import numpy as np
from nltk.tokenize import TweetTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

MM_EMBEDDINGS_DIM = 50
MM_HIDDEN_SIZE = 128
MM_MAX_DICT_SIZE = 100

TOKEN_UNK = "#unk"


class Model(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, stride=5),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, n_actions),
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 1),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


class ModelMultimodal(nn.Module):
    def __init__(self, input_shape, n_actions, max_dict_size=MM_MAX_DICT_SIZE):
        super(ModelMultimodal, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, stride=5),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.emb = nn.Embedding(max_dict_size, MM_EMBEDDINGS_DIM)
        self.rnn = nn.LSTM(MM_EMBEDDINGS_DIM, MM_HIDDEN_SIZE, batch_first=True)

        self.policy = nn.Sequential(
            nn.Linear(conv_out_size + MM_HIDDEN_SIZE*2, n_actions),
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size + MM_HIDDEN_SIZE*2, 1),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def _concat_features(self, img_out, rnn_hidden):
        batch_size = img_out.size()[0]
        if isinstance(rnn_hidden, tuple):
            flat_h = list(map(lambda t: t.view(batch_size, -1), rnn_hidden))
            rnn_h = torch.cat(flat_h, dim=1)
        else:
            rnn_h = rnn_hidden.view(batch_size, -1)
        return torch.cat((img_out, rnn_h), dim=1)

    def forward(self, x):
        x_img, x_text = x
        assert isinstance(x_text, rnn_utils.PackedSequence)

        # deal with text data
        emb_out = self.emb(x_text.data)
        emb_out_seq = rnn_utils.PackedSequence(emb_out, x_text.batch_sizes)
        rnn_out, rnn_h = self.rnn(emb_out_seq)

        # extract image features
        fx = x_img.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)

        feats = self._concat_features(conv_out, rnn_h)
        return self.policy(feats), self.value(feats)


class MultimodalPreprocessor:
    log = logging.getLogger("MulitmodalPreprocessor")

    def __init__(self, max_dict_size=MM_MAX_DICT_SIZE, device="cpu"):
        self.max_dict_size = max_dict_size
        self.token_to_id = {TOKEN_UNK: 0}
        self.next_id = 1
        self.tokenizer = TweetTokenizer(preserve_case=True)
        self.device = device

    def __len__(self):
        return len(self.token_to_id)

    def __call__(self, batch):
        """
        Convert list of multimodel observations (tuples with image and text string) into the form suitable
        for ModelMultimodal to disgest
        :param batch:
        """
        tokens_batch = []
        for img_obs, txt_obs in batch:
            tokens = self.tokenizer.tokenize(txt_obs)
            idx_obs = self.tokens_to_idx(tokens)
            tokens_batch.append((img_obs, idx_obs))
        # sort batch decreasing to seq len
        tokens_batch.sort(key=lambda p: len(p[1]), reverse=True)
        img_batch, seq_batch = zip(*tokens_batch)
        lens = list(map(len, seq_batch))

        # convert data into the target form
        # images
        img_v = torch.FloatTensor(img_batch).to(self.device)
        # sequences
        seq_arr = np.zeros(shape=(len(seq_batch), max(len(seq_batch[0]), 1)), dtype=np.int64)
        for idx, seq in enumerate(seq_batch):
            seq_arr[idx, :len(seq)] = seq
            # Map empty sequences into single #UNK token
            if len(seq) == 0:
                lens[idx] = 1
        seq_v = torch.LongTensor(seq_arr).to(self.device)
        seq_p = rnn_utils.pack_padded_sequence(seq_v, lens, batch_first=True)
        return img_v, seq_p

    def tokens_to_idx(self, tokens):
        res = []
        for token in tokens:
            idx = self.token_to_id.get(token)
            if idx is None:
                if self.next_id == self.max_dict_size:
                    self.log.warning("Maximum size of dict reached, token '%s' converted to #UNK token", token)
                    idx = 0
                else:
                    idx = self.next_id
                    self.next_id += 1
                    self.token_to_id[token] = idx
            res.append(idx)
        return res

    def save(self, file_name):
        with open(file_name, 'wb') as fd:
            pickle.dump(self.token_to_id, fd)
            pickle.dump(self.max_dict_size, fd)
            pickle.dump(self.next_id, fd)

    @classmethod
    def load(cls, file_name):
        with open(file_name, "rb") as fd:
            token_to_id = pickle.load(fd)
            max_dict_size = pickle.load(fd)
            next_id = pickle.load(fd)

            res = MultimodalPreprocessor(max_dict_size)
            res.token_to_id = token_to_id
            res.next_id = next_id
            return res


def train_demo(net, optimizer, batch, writer, step_idx, preprocessor, device="cpu"):
    """
    Train net on demonstration batch
    """
    batch_obs, batch_act = zip(*batch)
    batch_v = preprocessor(batch_obs).to(device)
    optimizer.zero_grad()
    ref_actions_v = torch.LongTensor(batch_act).to(device)
    policy_v = net(batch_v)[0]
    loss_v = F.cross_entropy(policy_v, ref_actions_v)
    loss_v.backward()
    optimizer.step()
    writer.add_scalar("demo_loss", loss_v.item(), step_idx)

