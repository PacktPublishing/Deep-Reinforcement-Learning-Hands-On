# Params tuning of a2c

## Initial version

```text
LEARNING_RATE = 0.001
ADAM_EPS = 1e-3
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1
```

Run Dec11_23-46-06_gpu-pong-a2c_t1, 
9M steps, 3 hours

Tweak one parameter at a time

## Larger LR

* LR = 0.002: converged in 4.8M, 1.5 hours
* LR = 0.003: converged in 3.6M, 1 hours
* LR = 0.004: no convergence
* LR = 0.005: no convergence

Good value is something between 0.002 and 0.003

## Entropy Beta

* Beta = 0.02: 6.8M steps, 2 hours (better)
* Beta = 0.03: 12M, 4 hours (worse)

## Environments count

* Envs = 40: 8.6M, 3h
* Envs = 30: 6.2M, 2h (looks like a lucky seed)
* Envs = 20: 9.5M, 3h
* Envs = 10: hasn't converged in 12M frames and 4.5h, stopped
* Envs = 60: 11.6M, 4H (looks like an unlucky seed)
* Envs = 70: 7.7M, 2.5H 

## Batch size

* Batch = 64: 4.9M, 1.7h
* Batch = 32: 3.8M, 1.5h (!!!)
* Batch = 16: Doesn't converge
* Batch = 8: Doesn't converge
