speed experiments:

with full loss: 11 f/s
without loss: 750 f/s
loss without backward and opt: 12.5 f/s
without bellman loop: 12.9 f/s
without tgt net, but with var: 12.0 f/s
without tgt net and no tgt var: 21 f/s

full, but async cuda in loss: 11.8 f/s 
