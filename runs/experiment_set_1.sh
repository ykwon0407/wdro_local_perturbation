#!/bin/bash

cd ~/deep/project/dro-id/

common_args="--nepoch 100"
#Run data with CUDA 0~1
for seed in 1 2 3 4 5; do
    for size in 1000 2500 5000 25000 50000; do
    CUDA_VISIBLE_DEVICES=0 python3 erm.py --train_dir experiments/erm10@${size} --dataset=cifar10.${seed}@${size}-1 --wd=0.02 --smoothing 0.001 $common_args &
    CUDA_VISIBLE_DEVICES=1 python3 erm.py --train_dir experiments/erm100@${size} --dataset=cifar100.${seed}@${size}-1 --wd=0.02 --smoothing 0.001 $common_args &
    CUDA_VISIBLE_DEVICES=2 python3 mixup.py --train_dir experiments/mixup10@${size} --dataset=cifar10.${seed}@${size}-1 $common_args &
    CUDA_VISIBLE_DEVICES=3 python3 mixup.py --train_dir experiments/mixup100@${size} --dataset=cifar100.${seed}@${size}-1 $common_args
    done
done

for seed in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=0 python3 erm.py --train_dir experiments/erm10@100 --dataset=cifar10.${seed}@100-1 --wd=0.02 --smoothing 0.001 $common_args &
    CUDA_VISIBLE_DEVICES=1 python3 mixup.py --train_dir experiments/mixup10@100 --dataset=cifar10.${seed}@100-1 $common_args &
    CUDA_VISIBLE_DEVICES=2 python3 erm.py --train_dir experiments/erm10@500 --dataset=cifar10.${seed}@500-1 --wd=0.02 --smoothing 0.001 $common_args &
    CUDA_VISIBLE_DEVICES=3 python3 mixup.py --train_dir experiments/mixup10@500 --dataset=cifar10.${seed}@500-1 $common_args 
done
