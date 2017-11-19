# 2017-11-16

Point-1 experiments: return back to the convergence state.

Data filtering was returned by default. Run started.

Convergence is great: reached positive reward in an hour.

Percentage reward scheme was implemented and run as Nov16_13-30-57_gpu-conv-p1-perc-reward

This scan is also converging, trying to run the same on all 2016 data.
Fixed the bug with extensive filtering on small quotes. 2016 data started Nov16_15-09-10_gpu-conv-p1-perc-reward-2016

Implemented next adjustment: bar open price tweak. If open price for current bar doesn't match close bar of the previous,
open price set to close bar. This should make more correct rewards.
Started as Nov16_16-42-59_gpu-conv-p1-open-fix-YNDX 

Started model with 20 bars context and 1m replay buffer on 2016 prices.
Started as Nov16_22-17-33_home-conv-p1-bars=20-replay=1m-2016

Need to implement good testing utility + validation test during training.

Results on current run is not very impressive: 2016 runs don't converge, run with fixed open price
converges much worse than before.

Need to implement:
1. [x] proper reward calculation which takes in account not only bar's movement, but also gaps. Disable open price fix and test.
1. [x] Volumes in the data
2. Testing of system on arbitrary data. Things need to be calculated:
    1. count of orders done
    2. ratio of profitable and loosing orders
    3. mean profitable and loosing order
    4. final balance
    5. max drawdown
    6. balance chart
    7. profit chart
    
Reward calculation implemented, run Nov17_08-55-51_gpu-conv-fix-reward-YNDX16   

Volumes implemented, run Nov17_10-15-13_gpu-conv-vols-YNDX16

Reward calculation stopped, works good. 

Implemented tracking of average episode length, as testing on YNDX16 has shown that the agent has tendency to open 
position and keep it (which is logical on growing market).

Started both volumes + average episode len.

Implemented validation on train and validation datasets.
Started as Nov17_13-11-30_gpu-conv-vols-val-YNDX16

Next run uses longer epsilon decay: 3M steps, compare convergence on yandex. Run name longer-epsilon-YNDX16

Need to figure out why bars=10 doesn't converge anymore. Suspects:
1. Percentage reward: 373b36fa04344bd5556e06e2954dd6a349e42800 (Nov19_10-46-16_gpu-conv-p2-bars=10-perc-reward)
2. filtering threshold lowered to 1e-8 ae3a0c8f0534da03f06f8b0f4888b9fa9c608fa1 (Nov19_10-55-54_gpu-conv-p2-bars=10-filtering)
3. proper reward for bar calculation 3e7b25ed503899745a07acab59f713323e4a9cf9
