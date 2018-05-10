#!/usr/bin/env python3
import subprocess
import random

LEARNING_RATE_EXP = (-5, -2)
NOISE_STD_EXP = (-5, -0.5)
ITERS = 50

if __name__ == "__main__":
    random.seed(1237)
    while True:
        lr_exp = random.uniform(LEARNING_RATE_EXP[0], LEARNING_RATE_EXP[1])
        std_exp = random.uniform(NOISE_STD_EXP[0], NOISE_STD_EXP[1])
        lr = 10 ** lr_exp
        std = 10 ** std_exp
        print(lr_exp, std_exp)
        print(lr, std)
        res = subprocess.run(["./02_breakout_es.py", "--cuda", "--lr", str(lr), '--noise-std', str(std),
                              "--iters", str(ITERS)])
    pass
