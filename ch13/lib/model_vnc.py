import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 2, stride=1),
        )

    def forward(self, x):
        return self.conv(x)
