# 2017-10-27
Breakout-v4 not converging. Maybe, it's replay buffer size.

Ways to overcome:
1. Try params from DQN paper: 1M buffer, 10k sync. Started on gpu#1, log-basic-full-1.txt, 
stopped, as 1M buffer will require 100GB of RAM
2. Change back to pong, the only difference is final epsilon - 10%
 
