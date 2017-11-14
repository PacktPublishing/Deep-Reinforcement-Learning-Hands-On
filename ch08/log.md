# 2017-11-12

Get max speed:
1. 128 batch -  84 f/s
2. 32 batch  - 141 f/s

Filter empty bars. Run "Nov13_07-21-51_gpu-simple-filter"
Still no convergence.

Next things to check:
* Disable noisy nets, replacing by long epsilon decay (1M frames or something)
* Give a reward after order close equal to full profit 

Both started.

Not clear why value of held-out states are dropping sharply after step 100k.
Maybe that's due to sampling of states which is performed not uniformly but with prio replay buffer...
Check this with simple replay buffer replacement...

Idea: implement simple growing trend (maybe with some noice) and test the agent on it.

# 2017-11-14

Reached some slow convergence. Currently running variants:
1. Nov13_12-33-18_gpu-simple-e-greedy: prioritized replay buffer, but with epsilon-greedy policy (noisy networks disabled). 56a7d8ed03c640b7ee7b6584c89c472f049bf790 
1. Nov13_12-54-38_gpu-simple-close-reward: prio replay buffer, but environment gives full position reward on position close. 2d9c861c47929dcf3b7432815c6f3600a7b2c08b
1. Nov13_14-18-18_home-simple-replay-simple: simple replay buffer. 42da48493152628e4dab9e2e394c12b14e1fd638

Results:
1. Slow convergence, which is weird. Maybe, I should try other values for alpha and large buffer. Or, maybe, my prio buffer is buggy :)
2. Better convergence, in 12m steps reached positive values, reward_100 ~= -1.0
3. The best convergence, got in positive area of reward_100 in 7m steps, 12m step has reward_100 +1

Next actions:

* Change trading logic to open position on the current bar, rather than on the next bar's close.
* Implement pretrain buffer with artificially-made data for one-two step order

  
