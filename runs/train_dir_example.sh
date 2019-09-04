#!/bin/bash

cd ~/deep/project/dro-id/


#train_dir: directory to save the model folder (such as erm_ema, l2_0.0001, maxsup_2, mixup ...)
#default train dir is ./experiments/
#Directory is saved as os.path.join(train_dir, model_dir with hyperparameters, dataset.${seed}@${size}-1,
## -example: ./experiments/l2_0.0001/cifar100.1@50000-1/tf, args  

#number of epochs
$common_args = --nepoch 100 
#erm
CUDA_VISIBLE_DEVICES=0 python3 erm.py --dataset=cifar10.${seed}@${size}-1 --wd=0.02 --smoothing 0.001 $common_args
#
CUDA_VISIBLE_DEVICES=0 python3 erm.py --dataset=cifar10.${seed}@${size}-1 --wd=0.02 --smoothing 0.001 $common_args


#Run data with CUDA 0~1
for seed in 1 2 3 4 5; do
    for size in 1000 2500 5000 25000 50000; do
    
    CUDA_VISIBLE_DEVICES=1 python3 erm.py --train_dir experiments/erm100@${size} --dataset=cifar100.${seed}@${size}-1 --wd=0.02 --smoothing 0.001 $common_args &
    CUDA_VISIBLE_DEVICES=2 python3 mixup.py --train_dir experiments/mixup10@${size} --dataset=cifar10.${seed}@${size}-1 $common_args &
    CUDA_VISIBLE_DEVICES=3 python3 mixup.py --train_dir experiments/mixup100@${size} --dataset=cifar100.${seed}@${size}-1 $common_args
    done
done
