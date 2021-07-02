#!/bin/bash

for lr in 0.001 0.0001
do
for lambda_l2 in 0.00001 0.0
do
python  train.py  -t simulated -lr $lr -lambda_l2 $lambda_l2
python  train.py  -t mimic -lr $lr -lambda_l2 $lambda_l2
python  train.py  -t stackoverflow -lr $lr -lambda_l2 $lambda_l2
python  train.py  -t retweet -lr $lr -lambda_l2 $lambda_l2
done
done