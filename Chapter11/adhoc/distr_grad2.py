#!/usr/bin/env python3
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp

import ptan


TRAIN_DATA = [1, 2, 3, 4, 5, 6]
CUDA = True


def get_y(x):
    return 2.5 * x + 10


def grad_fun(net, queue):
    iter_idx = 0
    while True:
        sum_loss = 0.0
        iter_idx += 1
        for v in TRAIN_DATA:
            x_v = Variable(torch.from_numpy(np.array([v], dtype=np.float32)))
            y_v = Variable(torch.from_numpy(np.array([get_y(v)], dtype=np.float32)))
            if CUDA:
                x_v = x_v.cuda()
                y_v = y_v.cuda()

            net.zero_grad()
            out_v = net(x_v)
            loss_v = F.mse_loss(out_v, y_v)
            loss_v.backward()

            grads = [param.grad.clone() if param.grad is not None else None
                     for param in net.parameters()]

            queue.put(grads)
            sum_loss += loss_v.data.cpu().numpy()
        print("%d: %.2f" % (iter_idx, sum_loss))
        if sum_loss < 0.1:
            queue.put(None)
            break


if __name__ == "__main__":
    mp.set_start_method('spawn')
    net = nn.Linear(in_features=1, out_features=1)
    if CUDA:
        net.cuda()
    tgt_net = ptan.agent.TargetNet(net)
    tgt_net.target_model.share_memory()
    print(net)

    for p in net.parameters():
        p.data.zero_()
    tgt_net.sync()

    optimizer = optim.Adam(net.parameters(), lr=1e-2)
    queue = mp.Queue(maxsize=10)
    grad_proc = mp.Process(target=grad_fun, args=(tgt_net.target_model, queue))
    grad_proc.start()

    iter_idx = 0

    while True:
        iter_idx += 1

        grads = queue.get()
        if grads is None:
            break
        for grad, param in zip(grads, net.parameters()):
            # v = Variable(torch.from_numpy(grad))
            # if CUDA:
            #     v = v.cuda()
            param.grad = grad
        optimizer.step()
        tgt_net.sync()
    pass

