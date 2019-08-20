#!/bin/bash

cd ~/deep/project/dro-id/

#number of epochs
common_args="--nepoch 30"

#Run data with CUDA 0~1
#maxmax.py
for LH in 10 20 30; do
    for size in 1000; do
        CUDA_VISIBLE_DEVICES=0 python3 maxmax.py --train_dir experiments/maxmax${LH}_10@${size} --dataset=cifar10.1@${size}-1 --LH ${LH} $common_args &
        CUDA_VISIBLE_DEVICES=1 python3 maxmax.py --train_dir experiments/maxmax${LH}_100@${size} --dataset=cifar100.1@${size}-1 --LH ${LH} $common_args
    done
done
wait

#maxl2.py
for LH in 0.1 1 10; do
    for size in 1000; do
        CUDA_VISIBLE_DEVICES=0 python3 maxl2.py --train_dir experiments/maxl2${LH}_10@${size} --dataset=cifar10.1@${size}-1 --LH ${LH} $common_args &
        CUDA_VISIBLE_DEVICES=1 python3 maxl2.py --train_dir experiments/maxl2${LH}_100@${size} --dataset=cifar100.1@${size}-1 --LH ${LH} $common_args
    done
done
wait

#l2.py
for gamma in 0.0001 0.0005 0.001; do
    for size in 1000; do
        CUDA_VISIBLE_DEVICES=0 python3 l2.py --train_dir experiments/l2${gamma}_10@${size} --dataset=cifar10.1@${size}-1 --gamma ${gamma} $common_args &
        CUDA_VISIBLE_DEVICES=1 python3 l2.py --train_dir experiments/l2${gamma}_100@${size} --dataset=cifar100.1@${size}-1 --gamma ${gamma} $common_args
    done
done

