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
Maybe that's due to sampling of states...

Idea: implement simple growing trend (maybe with some noice) and test the agent on it.
