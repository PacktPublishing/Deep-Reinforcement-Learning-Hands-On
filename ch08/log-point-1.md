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
3. proper reward for bar calculation 3e7b25ed503899745a07acab59f713323e4a9cf9 (Nov19_10-57-25_gpu-conv-p2-bars=10-reward)

All of them are converging.

Do bisect:
Start 3e7b25ed503899745a07acab59f713323e4a9cf9
Stop 6c12332edc460f535f78088243f24f1e85e47965


Bisect gave weird results, try brute-force check. Full history range
```text
[ ] 2febc2245bb382fcabac8a96c233204854001fe2 Log
[ ] 6c12332edc460f535f78088243f24f1e85e47965 upd log
[ ] 50d7d978f3fe616a2fcaf15b4c78c5b3de332dc5 Upd
[ ] 7dad5c468961901c06bc4ab8609b92e5f9c31655 Upd
[ ] ddc31c14835050f8fe991e0c00da8ac818a30c1e Bars=10 issue
[ ] 8cff76fc1d4e0bd659ecb81cf938665c6986d499 Try two steps
[ ] f9a49f95e4f136ad647d895d86f97268a372a0f6 Logs
[ ] c8ac06936df360c367ff5c50343ddc3380670eb5 Log
[ ] 1d15fe6dc7349b0c72e737064f34e69fd1527508 20 bars context
[ ] e9f9f35098774465dce05638f0eda15026d9834c Larger final epsilon
[ ] 8869eb6931380866c3ce64ef2c310c6495848521 3M steps
[ ] f705f23551422d8527d228b1141e80a04956bdf6 Fix order profit
[ ] ca734a0ad58dabc72d4a210b6f7849eede61e4ce Use 3M instead of 10M
[ ] 142957089e3260a73a4071b0ecbe3b4c85ff42e1 Return 10M back
[ ] 124b1f3f888a34e364bfe68d9de74e270c020f47 Roll back 10M
[ ] 0b32fe2e25873020b2aa6871550d21a96dd3be4e Cleanups
[ ] be2acba862234140986c8315d79885572f7fdce5 Longer epsilon decay
[ ] d08fff85ad6d252872bcf2f50259ecf2feac9299 Change names a bit
[-] d5c0ab3655b1082f254a83fc5ded5fd24b36acde Small tweaks
[ ] 17b7478eafa0797a2cebfd0d2580a4a38c65732c Validation
[ ] b53183d4e5644dd48dfd585f2216b6b0e56b7167 Epsilon-greedy in model test
[ ] c51622d94d1311bbec720c98832a4a3db0151246 track steps
[ ] 3411c77a70aad517138eb1d6127494b092f67627 Disable reward on close
[-] 03aae9abfc448e1639587cb9e60c55eafb0a586e Fix data filtering broken by volumes
[ ] e91f33beea90fa0fa8ace2390b45d52e65bf0482 Make volumes switchable
[-] 7fc0beb5e4e07a87362a53816be9a57a2fb215cb Log update
[ ] 0f2bce0c68246eeebc64f99d1f6989766d3ec7c8 Expose volumes in Conv state
[ ] 39af713f6bdfc9ae202defc8045e9c69f6c5ced8 Load volumes
[+] 71432c94f41c8fc0f3aab9d5d1051dcaa653ae72 Fix tests
[ ] c6e33cba772f2eb08fc3442191c720feb5cd0017 Roll back params
[ ] e176f18a4495311bbf05d8615e5e1b3fceb17394 Log
[+] 3e7b25ed503899745a07acab59f713323e4a9cf9 Calculate reward for the bar properly
[ ] 927c149ebfb89d117b82e327439fe931b55f7d19 Disable price fix and update logs
[ ] d2aa6481911614796bddd141029ba3a0a031aa4b Tweak params
[ ] 78281f35a41b2f6f76cfc73e7a26f7dc16e38ce1 Fix for price
[+] ae3a0c8f0534da03f06f8b0f4888b9fa9c608fa1 Fix bug with filtering
[+] 373b36fa04344bd5556e06e2954dd6a349e42800 Percentage reward
[ ] b40402b9fa342321a2def99cfeb4e224a4216f0e Return data filtering
```

Looks like the reason is in volumes. Will try to disable volumes in the environment.
Started run Nov20_12-56-20_gpu-conv-p2-bars=10-no-vols. Upd: yes, that was the reason of bad convergence!
So, volumes are not good for conv version.

Now, next check: compare version with reward on every bar and version with final reward.
Reward on close run: Nov20_18-03-59_gpu-conv-bars=10-reward-on-close
Reward on every bar: Nov20_18-06-06_gpu-conv-bars=10-reward-every-bar 

Larger model converges not very good.
Trying 25 bars on YNDX and SBER.
Runs: Nov21_07-13-54_gpu-conv-bars=25-YNDX16 and Nov21_07-15-08_gpu-conv-bars=25-SBER16
