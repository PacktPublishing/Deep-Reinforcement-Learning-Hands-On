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

## Larger LR

* LR = 0.002
* LR = 0.003
* LR = 0.004
* LR = 0.005
