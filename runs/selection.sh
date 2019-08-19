#!/bin/bash

cd ~/deep/project/dro-id/

#Run data with CUDA 0~1
for gamma in 0.01 0.1 1 10; do
    for size in 1000 2500 5000 25000 50000; do
        common_args="--nepoch 20"
    CUDA_VISIBLE_DEVICES=0 python3 grad_regularization.py --train_dir experiments/grad_regularization10@${size}b${gamma} --dataset=cifar10.1@${size}-1 $common_args &
    CUDA_VISIBLE_DEVICES=1 python3 grad_regularization.py --train_dir experiments/grad_regularization100@${size}b${gamma} --dataset=cifar100.1@${size}-1 $common_args
    done
    for size in 100 500; do
        CUDA_VISIBLE_DEVICES=1 python3 grad_regularization.py --train_dir experiments/grad_regularization10@${size}b${gamma} --dataset=cifar10.1@${size}-1 $common_args
    done
    wait
done
