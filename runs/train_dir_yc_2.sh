#!/bin/bash

cd ~/deep/project/dro-id/

#train_dir: directory to save the model folder (such as erm_ema, l2_0.0001, maxsup_2, mixup ...)
#default train dir is ./experiments/

#'tf' and 'args' are saved at: os.path.join(train_dir, model_dir with hyperparameters, dataset.${seed}@${size}-1)
####example: ./experiments/l2_0.0001/cifar100.1@50000-1/tf, args  

#number of epochs
common_args="--nepoch 100"

for seed in 1 2 3 4 5; do
    for size in 100 500 1000 2500 5000 25000 50000; do
        # CUDA_VISIBLE_DEVICES=0 python3 erm.py --dataset=cifar10.${seed}@${size}-1 --wd=0.02 --smoothing 0.001 $common_args &
        # CUDA_VISIBLE_DEVICES=1 python3 mixup_grad.py --dataset=cifar10.${seed}@${size}-1 $common_args 
        CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --dataset=cifar10.${seed}@${size}-1 --regularizer l2 --gamma 0.002 $common_args
        # CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --dataset=cifar10.${seed}@${size}-1 --regularizer l2 --gamma 0.0005 $common_args 
    done
done

# for seed in 1 2 3 4 5; do
#     for size in 1000 2500 5000 25000 50000; do
#         # CUDA_VISIBLE_DEVICES=0 python3 erm.py --dataset=cifar100.${seed}@${size}-1 --wd=0.02 --smoothing 0.001 $common_args &
#         # CUDA_VISIBLE_DEVICES=1 python3 mixup_grad.py --dataset=cifar100.${seed}@${size}-1 $common_args 
#         CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --dataset=cifar100.${seed}@${size}-1 --regularizer l2 --gamma 0.00025 $common_args
#         # CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --dataset=cifar100.${seed}@${size}-1 --regularizer l2 --gamma 0.0005 $common_args 
#     done
# done

