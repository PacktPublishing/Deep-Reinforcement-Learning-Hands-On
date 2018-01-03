import torch
import torch.nn as nn


class PhraseModel(nn.Module):
    def __init__(self, emb_size, dict_size, hid_size):
        super(PhraseModel, self).__init__()

        self.encoder = nn.RNN(input_size=emb_size, hidden_size=hid_size, num_layers=1, batch_first=True)
        self.decoder = nn.RNN(input_size=emb_size, hidden_size=hid_size, num_layers=1, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hid_size, dict_size)
        )

    def forward(self, x):
        _, hid = self.encoder(x)
        return hid
