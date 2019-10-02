#!/bin/bash

cd ~/deep/project/dro-id/

#examples: salt and pepper with p=0.01
#CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --eval_ckpt experiments/l2_0.0002/cifar10.1@5000-1/tf/model.ckpt-06553600 -dataset=cifar10.1@5000-1 --regularizer l2 --gamma 0.0002 --noise_mode 's&p' --noise_p 0.01

#examples: gaussian with mean=0 and var=0.001
#CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --eval_ckpt experiments/l2_0.0002/cifar10.1@5000-1/tf/model.ckpt-06553600 -dataset=cifar10.1@5000-1 --regularizer l2 --gamma 0.0002 --noise_mode 'gaussian' --noise_mean 0 --noise_var 0.001
