#!/bin/bash

cd ~/deep/project/dro-id/


# CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py  --eval_ckpt experiments/test_mixup/cifar10.1@5000-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.0_filters32_gamma1.0_lr0.002_nclass10_regularizerNone_repeat4_scales3_wd0.002 --train_dir experiments/test_mixup --dataset=cifar10.1@5000-1 --nepoch 100 --regularizer None --ema 0.0 &
# CUDA_VISIBLE_DEVICES=1 python3 mixup_grad.py  --eval_ckpt experiments/test_mixup_ema/cifar10.1@5000-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass10_regularizerNone_repeat4_scales3_wd0.002 --train_dir experiments/test_mixup_ema --dataset=cifar10.1@5000-1 --nepoch 100 --regularizer None --ema 0.999

#After naming:
#CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py  --eval_ckpt experiments/mixup/cifar100.5@5000-1/tf/model.ckpt-00655360 --dataset=cifar100.5@5000-1 --regularizer None

for steps in 00655360 01310720 01966080 02621440 03276800 03932160 04587520 05242880 05898240 06553600; do
	for size in 100 500 1000 2500 5000 25000 50000; do
		for seed in 1 2 3 4 5; do
			CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py  --eval_ckpt experiments/l2_0.0001/cifar10.$seed@$size-1/tf/model.ckpt-$steps --dataset=cifar100.$seed@$size-1 --regularizer l2 --gamma 0.0001 &
			CUDA_VISIBLE_DEVICES=1 python3 mixup_grad.py  --eval_ckpt experiments/l2_0.0002/cifar10.$seed@$size-1/tf/model.ckpt-$steps --dataset=cifar100.$seed@$size-1 --regularizer l2 --gamma 0.0002
		done
	done
done

wait

for steps in 00655360 01310720 01966080 02621440 03276800 03932160 04587520 05242880 05898240 06553600; do
	for size in 1000 2500 5000 25000 50000; do
		for seed in 1 2 3 4 5; do
			CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py  --eval_ckpt experiments/l2_0.0001/cifar100.$seed@$size-1/tf/model.ckpt-$steps --dataset=cifar100.$seed@$size-1 --regularizer l2 --gamma 0.0001 &
			CUDA_VISIBLE_DEVICES=1 python3 mixup_grad.py  --eval_ckpt experiments/l2_0.0002/cifar100.$seed@$size-1/tf/model.ckpt-$steps --dataset=cifar100.$seed@$size-1 --regularizer l2 --gamma 0.0002
		done
	done
done
