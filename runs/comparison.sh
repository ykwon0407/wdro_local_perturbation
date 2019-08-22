#!/bin/bash

cd ~/deep/project/dro-id/

#number of epochs
common_args="--nepoch 100"

#Run data with CUDA 0~1
#maxmax.py
for LH in 0.5 1.5; do
    for size in 1000; do
        CUDA_VISIBLE_DEVICES=0 python3 grad_regularization.py --train_dir experiments/maxmax${LH}_10@${size} --dataset=cifar10.1@${size}-1 --LH ${LH} --regularizer maxsup $common_args
        #CUDA_VISIBLE_DEVICES=1 python3 grad_regularization.py --train_dir experiments/maxmax${LH}_100@${size} --dataset=cifar100.1@${size}-1 --LH ${LH} $common_args
    done
done &


#maxl2.py
for LH in 2 4; do
    for size in 1000; do
        CUDA_VISIBLE_DEVICES=1 python3 grad_regularization.py --train_dir experiments/maxl2${LH}_10@${size} --dataset=cifar10.1@${size}-1 --LH ${LH} --regularizer maxl2 $common_args
        #CUDA_VISIBLE_DEVICES=1 python3 grad_regularization.py --train_dir experiments/maxl2${LH}_100@${size} --dataset=cifar100.1@${size}-1 --LH ${LH} $common_args
    done
done
wait

#l2.py
for gamma in 1; do
    for size in 1000; do
        CUDA_VISIBLE_DEVICES=0 python3 grad_regularization.py --train_dir experiments/l2${gamma}_10@${size} --dataset=cifar10.1@${size}-1 --gamma ${gamma} $common_args --regularizer l2
        #CUDA_VISIBLE_DEVICES=1 python3 grad_regularization.py --train_dir experiments/l2${gamma}_100@${size} --dataset=cifar100.1@${size}-1 --gamma ${gamma} $common_args --regularizer l2
    done
done &

#l2.py
for gamma in 10 ; do
    for size in 1000; do
        CUDA_VISIBLE_DEVICES=1 python3 grad_regularization.py --train_dir experiments/l2${gamma}_10@${size} --dataset=cifar10.1@${size}-1 --gamma ${gamma} $common_args --regularizer l2
        #CUDA_VISIBLE_DEVICES=1 python3 grad_regularization.py --train_dir experiments/l2${gamma}_100@${size} --dataset=cifar100.1@${size}-1 --gamma ${gamma} $common_args --regularizer l2
    done
done
