#!/bin/bash

for lr in 0.001 0.0001
do
python  train.py  -t simulated $lr lr
python  train.py  -t mimic $lr lr
python  train.py  -t stackoverflow $lr lr
python  train.py  -t retweet $lr lr
done