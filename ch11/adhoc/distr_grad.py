#!/usr/bin/env python3
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import ptan


TRAIN_DATA = [1, 2, 3, 4, 5, 6]


def get_y(x):
    return 2.5 * x + 10


if __name__ == "__main__":
    net = nn.Linear(in_features=1, out_features=1)
    tgt_net = ptan.agent.TargetNet(net)
    print(net)

    for p in net.parameters():
        p.data.zero_()

    optimizer = optim.Adam(net.parameters(), lr=1e-2)
    iter_idx = 0

    while True:
        iter_idx += 1
        sum_loss = 0.0
        for v in TRAIN_DATA:
            tgt_net.sync()
            x_v = Variable(torch.from_numpy(np.array([v], dtype=np.float32)))
            y_v = Variable(torch.from_numpy(np.array([get_y(v)], dtype=np.float32)))

            tgt_net.target_model.zero_grad()
            out_v = tgt_net.target_model(x_v)
            loss_v = F.mse_loss(out_v, y_v)
            loss_v.backward()
            grads = [param.grad.data.cpu().numpy() if param.grad is not None else None
                     for param in tgt_net.target_model.parameters()]

            # apply gradients
            for grad, param in zip(grads, net.parameters()):
                param.grad = Variable(torch.from_numpy(grad))

            optimizer.step()
            sum_loss += loss_v.data.cpu().numpy()
        print("%d: %.2f" % (iter_idx, sum_loss))
        if sum_loss < 0.1:
            break

    pass

