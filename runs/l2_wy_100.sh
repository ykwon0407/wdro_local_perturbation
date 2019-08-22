#!/bin/bash

cd ~/deep/project/dro-id/
common_args="--nepoch 100 --regularizer l2"

# CIFAR 10
#for seed in 1 2 3 4 5; do
#    for size in 100 500 1000 2500 5000 25000 50000; do
#    CUDA_VISIBLE_DEVICES=1 python3 grad_regularization.py --train_dir experiments/l2_0.0005_10@${size} --dataset=cifar10.${seed}@${size}-1 --gamma 0.0005 $common_args
#    CUDA_VISIBLE_DEVICES=1 python3 grad_regularization.py --train_dir experiments/l2_0.001_10@${size} --dataset=cifar10.${seed}@${size}-1 --gamma 0.001 $common_args  
#    CUDA_VISIBLE_DEVICES=1 python3 grad_regularization.py --train_dir experiments/l2_0.0025_10@${size} --dataset=cifar10.${seed}@${size}-1 --gamma 0.0025 $common_args
#    done
#done

# CIFAR 100
for seed in 1 2 3 4 5; do
    for size in 1000 2500 5000 25000 50000; do
    CUDA_VISIBLE_DEVICES=0 python3 grad_regularization.py --train_dir experiments/l2_0.0005_100@${size} --dataset=cifar100.${seed}@${size}-1 --gamma 0.0005 $common_args
    CUDA_VISIBLE_DEVICES=0 python3 grad_regularization.py --train_dir experiments/l2_0.001_100@${size} --dataset=cifar100.${seed}@${size}-1 --gamma 0.001 $common_args 
    CUDA_VISIBLE_DEVICES=0 python3 grad_regularization.py --train_dir experiments/l2_0.0025_100@${size} --dataset=cifar100.${seed}@${size}-1 --gamma 0.0025 $common_args
    done
done
