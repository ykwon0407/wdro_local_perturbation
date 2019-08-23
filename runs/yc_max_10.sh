#!/bin/bash

cd ~/deep/project/dro-id/
common_args="--nepoch 100 --regularizer maxsup"

# CIFAR 10
for seed in 1 2 3 4 5; do
    for size in 100 500 1000 2500 5000 25000 50000; do
    CUDA_VISIBLE_DEVICES=0 python3 grad_regularization.py --train_dir experiments/max_4_10@${size} --dataset=cifar10.${seed}@${size}-1 --LH 4 $common_args &
    CUDA_VISIBLE_DEVICES=1 python3 grad_regularization.py --train_dir experiments/max_6_10@${size} --dataset=cifar10.${seed}@${size}-1 --LH 6 $common_args & 
    CUDA_VISIBLE_DEVICES=2 python3 grad_regularization.py --train_dir experiments/max_8_10@${size} --dataset=cifar10.${seed}@${size}-1 --LH 8 $common_args &
    CUDA_VISIBLE_DEVICES=3 python3 grad_regularization.py --train_dir experiments/max_10_10@${size} --dataset=cifar10.${seed}@${size}-1 --LH 10 $common_args 
    done
done
