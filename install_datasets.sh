#!/bin/bash

# Download datasets
CUDA_VISIBLE_DEVICES=3 python3 ./scripts/create_datasets.py
# cp $ML_DATA/svhn-test.tfrecord $ML_DATA/svhn_noextra-test.tfrecord

# Create semi-supervised subsets
for seed in 1 2 3 4 5; do
    for size in 250 500 1000 2000 4000; do
        # CUDA_VISIBLE_DEVICES=3 python3 scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
        # CUDA_VISIBLE_DEVICES=3 python3 scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL/svhn_noextra $ML_DATA/svhn-train.tfrecord &
        CUDA_VISIBLE_DEVICES=3 python3 scripts/create_split.py --seed=$seed --size=$size $ML_DATA/cifar10 $ML_DATA/cifar10-train.tfrecord &
    done
    # CUDA_VISIBLE_DEVICES=3 python3 scripts/create_split.py --seed=$seed --size=10000 $ML_DATA/SSL/cifar100 $ML_DATA/cifar100-train.tfrecord &
    wait
done

