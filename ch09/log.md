# 2017-12-05

No convergence. Gradient charts shows max gradient is ~0.06, which might be too small. Try to increase LR

Try: 
* LR=0.001: everything broke down, no exploration, etc. Trying to increase entropy beta
* LR=0.001, beta=0.01: helped with exploration, initial 300k steps hooks good. Maybe gradient clipping will be required.

Trying to remove target net.
