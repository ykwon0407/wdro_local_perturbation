#!/bin/bash

cd ~/deep/project/dro-id/

# Download CIFAR-10 and CIFAR-100 datasets
CUDA_VISIBLE_DEVICES=0 python3 ./input/create_datasets.py cifar10 cifar100

# Create semi-supervised subsets
for seed in 1 2 3 4 5; do
    for size in 1000 2500 5000 25000 50000; do
        CUDA_VISIBLE_DEVICES=0 python3 ./input/create_split.py --seed=$seed --size=$size ./input/cifar10/cifar10 ./input/cifar10-train.tfrecord &
        CUDA_VISIBLE_DEVICES=1 python3 ./input/create_split.py --seed=$seed --size=$size ./input/cifar100/cifar100 ./input/cifar100-train.tfrecord
    done
    for size in 100 500; do
        CUDA_VISIBLE_DEVICES=0 python3 ./input/create_split.py --seed=$seed --size=$size ./input/cifar10/cifar10 ./input/cifar10-train.tfrecord
    done
    wait
done


