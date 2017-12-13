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

* Envs = 40
* Envs = 30
* Envs = 20
* Envs = 10 
