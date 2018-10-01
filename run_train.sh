#!/bin/bash

python3 train.py \
  --data_dir=/home/jacobsuwang/Documents/UTA/Fall2018/LIN389C/GCN/Data/ \
  --data_name=citeseer \
  --hidden_size_1=50 \
  --learning_rate=1e-4 \
  --drop_prob=0.0 \
  --l2_coefficient=0.0 \
  --number_iterations=1000 \
  --print_every=50