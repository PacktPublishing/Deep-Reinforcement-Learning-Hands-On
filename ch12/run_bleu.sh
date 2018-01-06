#!/usr/bin/env bash
set -x

./train_scst.py --cuda --cornell western -n sc-corn-west-b_10-s_1 -l saves/xe-cor-west/pre_bleu_0.114_01.dat --samples 1
./train_scst.py --cuda --cornell western -n sc-corn-west-b_20-s_1 -l saves/xe-cor-west/pre_bleu_0.207_31.dat --samples 1
./train_scst.py --cuda --cornell western -n sc-corn-west-b_30-s_1 -l saves/xe-cor-west/pre_bleu_0.317_37.dat --samples 1
./train_scst.py --cuda --cornell western -n sc-corn-west-b_40-s_1 -l saves/xe-cor-west/pre_bleu_0.400_41.dat --samples 1
./train_scst.py --cuda --cornell western -n sc-corn-west-b_50-s_1 -l saves/xe-cor-west/pre_bleu_0.512_45.dat --samples 1
./train_scst.py --cuda --cornell western -n sc-corn-west-b_60-s_1 -l saves/xe-cor-west/pre_bleu_0.630_49.dat --samples 1
./train_scst.py --cuda --cornell western -n sc-corn-west-b_70-s_1 -l saves/xe-cor-west/pre_bleu_0.711_52.dat --samples 1
./train_scst.py --cuda --cornell western -n sc-corn-west-b_80-s_1 -l saves/xe-cor-west/pre_bleu_0.812_56.dat --samples 1
./train_scst.py --cuda --cornell western -n sc-corn-west-b_90-s_1 -l saves/xe-cor-west/pre_bleu_0.900_62.dat --samples 1
