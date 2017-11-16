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
