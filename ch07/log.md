# 2017-10-27
Breakout-v4 not converging. Maybe, it's replay buffer size.

Ways to overcome:
1. Try params from DQN paper: 1M buffer, 10k sync. Started on gpu#1, log-basic-full-1.txt, 
stopped, as 1M buffer will require 100GB of RAM
2. Change back to pong, the only difference is final epsilon - 10%

Started three versions:
1. basic
2. 2-steps
3. double dqn

Upd: convergence is weird, reverted epsilon back to 2% and restarted.

Next to implement: noisy nets, using noisy layer using https://github.com/Kaixhin/NoisyNet-A3C/blob/master/model.py
 
--Noisy layer in Kaixhin looks suspicious, need to check with noise turned off.--
Found the reason, original code assumed sample_noise call every iteration.
Rerun started, need to wait.

Todo:
1. run new version of noisy net, check convergence. Started as log-noisy-3.txt
2. implement factorized gaussian noise from the paper

# 2017-10-28

Noisy net converged. Still need to check factorized version, but so far I have done:
1. basic test
2. 2-step
3. double net
4. noisy net

Still need to be implemented and verified:
* Dueling version
* Prio replay buffer
* Distributional

Factorized noise v1 converges badly. Will try:
1. separate random var for bias (v2) -- not helped much
2. fix initialization and f(x) (v3) -- not converging
2. remove requires_grad=False from vars (v4)
3. roll back to full noise

Need to bring initialization to independent noise and test it.
