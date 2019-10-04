#!/bin/bash

cd ~/deep/project/dro-id/

#examples: salt and pepper with p=0.01
#CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --eval_ckpt experiments/l2_0.0002/cifar10.1@5000-1/tf/model.ckpt-06553600 -dataset=cifar10.1@5000-1 --regularizer l2 --gamma 0.0002 --noise_mode 's&p' --noise_p 0.01

#examples: gaussian with mean=0 and var=0.001
#CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --eval_ckpt experiments/l2_0.0002/cifar10.1@5000-1/tf/model.ckpt-06553600 -dataset=cifar10.1@5000-1 --regularizer l2 --gamma 0.0002 --noise_mode 'gaussian' --noise_mean 0 --noise_var 0.001

for p in 0.005 0.01 0.02; do
	for size in 100 500 1000 2500 5000 25000 50000; do
		for seed in 1 2 3 4 5; do		
			CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --eval_ckpt experiments/mixup/cifar10.$seed@$size-1/tf/model.ckpt-06553600 -dataset=cifar10.$seed@$size-1 --noise_p $p --regularizer='None' --noise_mode 's&p'
		done
	done
done

for p in 0.005 0.01 0.02; do
	for size in 1000 2500 5000 25000 50000; do
		for seed in 1 2 3 4 5; do		
			CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --eval_ckpt experiments/mixup/cifar100.$seed@$size-1/tf/model.ckpt-06553600 -dataset=cifar100.$seed@$size-1 --noise_p $p --regularizer='None' --noise_mode 's&p'
		done
	done
done

for p in 0.005 0.01 0.02; do
	for size in 100 500 1000 2500 5000 25000 50000; do
		for seed in 1 2 3 4 5; do		
			CUDA_VISIBLE_DEVICES=0 python3 erm.py --eval_ckpt experiments/erm/cifar10.$seed@$size-1/tf/model.ckpt-06553600 -dataset=cifar10.$seed@$size-1 --noise_mode 's&p' --noise_p $p
		done
	done
done

for p in 0.005 0.01 0.02; do
	for size in 1000 2500 5000 25000 50000; do
		for seed in 1 2 3 4 5; do		
			CUDA_VISIBLE_DEVICES=0 python3 erm.py --eval_ckpt experiments/erm/cifar100.$seed@$size-1/tf/model.ckpt-06553600 -dataset=cifar100.$seed@$size-1 --noise_mode 's&p' --noise_p $p
		done
	done
done

for gamma in 0.0001 0.0002; do
	for p in 0.005 0.01 0.02; do
		for size in 100 500 1000 2500 5000 25000 50000; do
			for seed in 1 2 3 4 5; do		
				CUDA_VISIBLE_DEVICES=1 python3 mixup_grad.py --eval_ckpt experiments/l2_$gamma/cifar10.$seed@$size-1/tf/model.ckpt-06553600 -dataset=cifar10.$seed@$size-1 --gamma $gamma  --noise_p $p --regularizer l2 --noise_mode 's&p'
			done
		done
	done
done

for gamma in 0.0001 0.0002; do
	for p in 0.005 0.01 0.02; do
		for size in 1000 2500 5000 25000 50000; do
			for seed in 1 2 3 4 5; do		
				CUDA_VISIBLE_DEVICES=1 python3 mixup_grad.py --eval_ckpt experiments/l2_$gamma/cifar100.$seed@$size-1/tf/model.ckpt-06553600 -dataset=cifar100.$seed@$size-1 --gamma $gamma  --noise_p $p --regularizer l2 --noise_mode 's&p'
			done
		done
	done
done
