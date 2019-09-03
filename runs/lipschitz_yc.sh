#!/bin/bash

cd ~/deep/project/dro-id/

# common_args="--regularizer maxsup"
# # CIFAR 10
# for seed in 1 2 3 4 5; do
# 	for size in 100 500 1000 2500 5000 25000 50000; do
# 		CUDA_VISIBLE_DEVICES=0 python3 grad_regularization.py --LH 4 --eval_ckpt experiments/max_4_10@${size}/cifar10.${seed}@${size}-1/FSgradient_LH4.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass10_regularizermaxsup_repeat4_scales3_wd0.002 --train_dir experiments/max_4_10@${size} --dataset=cifar10.${seed}@${size}-1 $common_args
# 		CUDA_VISIBLE_DEVICES=1 python3 grad_regularization.py --LH 6 --eval_ckpt experiments/max_6_10@${size}/cifar10.${seed}@${size}-1/FSgradient_LH6.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass10_regularizermaxsup_repeat4_scales3_wd0.002 --train_dir experiments/max_6_10@${size} --dataset=cifar10.${seed}@${size}-1 $common_args
# 		CUDA_VISIBLE_DEVICES=2 python3 grad_regularization.py --LH 8 --eval_ckpt experiments/max_8_10@${size}/cifar10.${seed}@${size}-1/FSgradient_LH8.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass10_regularizermaxsup_repeat4_scales3_wd0.002 --train_dir experiments/max_8_10@${size} --dataset=cifar10.${seed}@${size}-1 $common_args
# 		CUDA_VISIBLE_DEVICES=3 python3 grad_regularization.py --LH 10 --eval_ckpt experiments/max_10_10@${size}/cifar10.${seed}@${size}-1/FSgradient_LH10.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass10_regularizermaxsup_repeat4_scales3_wd0.002 --train_dir experiments/max_10_10@${size} --dataset=cifar10.${seed}@${size}-1 $common_args
# 	done
# done

# # CIFAR 100
# for seed in 1 2 3 4 5; do
# 	for size in 1000 2500 5000 25000 50000; do
# 		CUDA_VISIBLE_DEVICES=3 python3 grad_regularization.py --LH 4 --eval_ckpt experiments/max_4_100@${size}/cifar100.${seed}@${size}-1/FSgradient_LH4.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass100_regularizermaxsup_repeat4_scales3_wd0.002 --train_dir experiments/max_4_100@${size} --dataset=cifar100.${seed}@${size}-1 $common_args
# 		CUDA_VISIBLE_DEVICES=3 python3 grad_regularization.py --LH 6 --eval_ckpt experiments/max_6_100@${size}/cifar100.${seed}@${size}-1/FSgradient_LH6.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass100_regularizermaxsup_repeat4_scales3_wd0.002 --train_dir experiments/max_6_100@${size} --dataset=cifar100.${seed}@${size}-1 $common_args
# 		CUDA_VISIBLE_DEVICES=3 python3 grad_regularization.py --LH 8 --eval_ckpt experiments/max_8_100@${size}/cifar100.${seed}@${size}-1/FSgradient_LH8.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass100_regularizermaxsup_repeat4_scales3_wd0.002 --train_dir experiments/max_8_100@${size} --dataset=cifar100.${seed}@${size}-1 $common_args
# 		CUDA_VISIBLE_DEVICES=3 python3 grad_regularization.py --LH 10 --eval_ckpt experiments/max_10_100@${size}/cifar100.${seed}@${size}-1/FSgradient_LH10.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass100_regularizermaxsup_repeat4_scales3_wd0.002 --train_dir experiments/max_10_100@${size} --dataset=cifar100.${seed}@${size}-1 $common_args
# 	done
# done



common_args="--regularizer None"
# CIFAR 10
for seed in 1 2 3 4 5; do
	for size in 100 500 1000 2500 5000 25000 50000; do
		CUDA_VISIBLE_DEVICES=1 python3 grad_regularization.py $common_args --eval_ckpt /data/dro-id_history/mixup10@${size}/cifar10.${seed}@${size}-1/FSMixup_archresnet_batch64_beta0.5_ema0.999_filters32_lr0.002_nclass10_repeat4_scales3_wd0.002 --train_dir /data/dro-id_history/mixup10@${size} --dataset=cifar10.${seed}@${size}-1
	done
done

# CIFAR 100
for seed in 1 2 3 4 5; do
	for size in 1000 2500 5000 25000 50000; do
		CUDA_VISIBLE_DEVICES=2 python3 grad_regularization.py $common_args --eval_ckpt /data/dro-id_history/mixup100@${size}/cifar100.${seed}@${size}-1/FSMixup_archresnet_batch64_beta0.5_ema0.999_filters32_lr0.002_nclass100_repeat4_scales3_wd0.002 --train_dir /data/dro-id_history/mixup100@${size} --dataset=cifar100.${seed}@${size}-1
	done
done













