#!/bin/bash

cd ~/deep/project/dro-id/

#train_dir: directory to save the model folder (such as erm_ema, l2_0.0001, maxsup_2, mixup ...)
#default train dir is ./experiments/

#'tf' and 'args' are saved at: os.path.join(train_dir, model_dir with hyperparameters, dataset.${seed}@${size}-1)
####example: ./experiments/l2_0.0001/cifar100.1@50000-1/tf, args  

#number of epochs
$common_args = --nepoch 100 

#erm
CUDA_VISIBLE_DEVICES=0 python3 erm.py --dataset=cifar10.1@50000-1 --wd=0.02 --smoothing 0.001 $common_args
#### saved at ./experiments/erm/cifar10.1@50000-1/tf, args

#mixup
CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --dataset=cifar100.5@5000-1 $common_args
#### saved at ./experiments/mixup/cifar100.5@5000-1/tf, args

#l2
CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --dataset=cifar100.5@5000-1 --regularizer l2 --gamma 0.1234 $common_args
#### saved at ./experiments/l2_0.1234/cifar100.5@5000-1/tf, args

