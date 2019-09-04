#!/bin/bash

cd ~/deep/project/dro-id/


# CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py  --eval_ckpt experiments/test_mixup/cifar10.1@5000-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.0_filters32_gamma1.0_lr0.002_nclass10_regularizerNone_repeat4_scales3_wd0.002 --train_dir experiments/test_mixup --dataset=cifar10.1@5000-1 --nepoch 100 --regularizer None --ema 0.0 &
# CUDA_VISIBLE_DEVICES=1 python3 mixup_grad.py  --eval_ckpt experiments/test_mixup_ema/cifar10.1@5000-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass10_regularizerNone_repeat4_scales3_wd0.002 --train_dir experiments/test_mixup_ema --dataset=cifar10.1@5000-1 --nepoch 100 --regularizer None --ema 0.999

# for size in 5000 50000; do
# 	CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py  --eval_ckpt experiments/test_mixup_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass10_regularizerNone_repeat4_scales3_wd0.002/tf/model.ckpt-00655360 --train_dir experiments/test_mixup_ema --dataset=cifar10.1@${size}-1 --regularizer None &
# 	CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py  --eval_ckpt experiments/test_mixup_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass10_regularizerNone_repeat4_scales3_wd0.002/tf/model.ckpt-01310720 --train_dir experiments/test_mixup_ema --dataset=cifar10.1@${size}-1 --regularizer None &
# 	CUDA_VISIBLE_DEVICES=1 python3 mixup_grad.py  --eval_ckpt experiments/test_mixup_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass10_regularizerNone_repeat4_scales3_wd0.002/tf/model.ckpt-01966080 --train_dir experiments/test_mixup_ema --dataset=cifar10.1@${size}-1 --regularizer None &
# 	CUDA_VISIBLE_DEVICES=1 python3 mixup_grad.py  --eval_ckpt experiments/test_mixup_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass10_regularizerNone_repeat4_scales3_wd0.002/tf/model.ckpt-02621440 --train_dir experiments/test_mixup_ema --dataset=cifar10.1@${size}-1 --regularizer None &
# 	CUDA_VISIBLE_DEVICES=1 python3 mixup_grad.py  --eval_ckpt experiments/test_mixup_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass10_regularizerNone_repeat4_scales3_wd0.002/tf/model.ckpt-03276800 --train_dir experiments/test_mixup_ema --dataset=cifar10.1@${size}-1 --regularizer None &
# 	CUDA_VISIBLE_DEVICES=2 python3 mixup_grad.py  --eval_ckpt experiments/test_mixup_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass10_regularizerNone_repeat4_scales3_wd0.002/tf/model.ckpt-03932160 --train_dir experiments/test_mixup_ema --dataset=cifar10.1@${size}-1 --regularizer None &
# 	CUDA_VISIBLE_DEVICES=2 python3 mixup_grad.py  --eval_ckpt experiments/test_mixup_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass10_regularizerNone_repeat4_scales3_wd0.002/tf/model.ckpt-04587520 --train_dir experiments/test_mixup_ema --dataset=cifar10.1@${size}-1 --regularizer None &
# 	CUDA_VISIBLE_DEVICES=2 python3 mixup_grad.py  --eval_ckpt experiments/test_mixup_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass10_regularizerNone_repeat4_scales3_wd0.002/tf/model.ckpt-05242880 --train_dir experiments/test_mixup_ema --dataset=cifar10.1@${size}-1 --regularizer None &
# 	CUDA_VISIBLE_DEVICES=3 python3 mixup_grad.py  --eval_ckpt experiments/test_mixup_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass10_regularizerNone_repeat4_scales3_wd0.002/tf/model.ckpt-05898240 --train_dir experiments/test_mixup_ema --dataset=cifar10.1@${size}-1 --regularizer None &
# 	CUDA_VISIBLE_DEVICES=3 python3 mixup_grad.py  --eval_ckpt experiments/test_mixup_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma1.0_lr0.002_nclass10_regularizerNone_repeat4_scales3_wd0.002/tf/model.ckpt-06553600 --train_dir experiments/test_mixup_ema --dataset=cifar10.1@${size}-1 --regularizer None 
# done

for size in 5000 50000; do
	CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py  --eval_ckpt experiments/test_l2_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma0.0025_lr0.002_nclass10_regularizerl2_repeat4_scales3_wd0.002/tf/model.ckpt-00655360 --train_dir experiments/test_l2_ema --dataset=cifar10.1@${size}-1 --nepoch 100 --regularizer l2 --gamma 0.0025 &
	CUDA_VISIBLE_DEVICES=1 python3 mixup_grad.py  --eval_ckpt experiments/test_l2_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma0.0025_lr0.002_nclass10_regularizerl2_repeat4_scales3_wd0.002/tf/model.ckpt-01310720 --train_dir experiments/test_l2_ema --dataset=cifar10.1@${size}-1 --nepoch 100 --regularizer l2 --gamma 0.0025 &
	CUDA_VISIBLE_DEVICES=1 python3 mixup_grad.py  --eval_ckpt experiments/test_l2_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma0.0025_lr0.002_nclass10_regularizerl2_repeat4_scales3_wd0.002/tf/model.ckpt-01966080 --train_dir experiments/test_l2_ema --dataset=cifar10.1@${size}-1 --nepoch 100 --regularizer l2 --gamma 0.0025 &
	CUDA_VISIBLE_DEVICES=1 python3 mixup_grad.py  --eval_ckpt experiments/test_l2_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma0.0025_lr0.002_nclass10_regularizerl2_repeat4_scales3_wd0.002/tf/model.ckpt-02621440 --train_dir experiments/test_l2_ema --dataset=cifar10.1@${size}-1 --nepoch 100 --regularizer l2 --gamma 0.0025 &
	CUDA_VISIBLE_DEVICES=2 python3 mixup_grad.py  --eval_ckpt experiments/test_l2_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma0.0025_lr0.002_nclass10_regularizerl2_repeat4_scales3_wd0.002/tf/model.ckpt-03276800 --train_dir experiments/test_l2_ema --dataset=cifar10.1@${size}-1 --nepoch 100 --regularizer l2 --gamma 0.0025 &
	CUDA_VISIBLE_DEVICES=2 python3 mixup_grad.py  --eval_ckpt experiments/test_l2_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma0.0025_lr0.002_nclass10_regularizerl2_repeat4_scales3_wd0.002/tf/model.ckpt-03932160 --train_dir experiments/test_l2_ema --dataset=cifar10.1@${size}-1 --nepoch 100 --regularizer l2 --gamma 0.0025 &
	CUDA_VISIBLE_DEVICES=2 python3 mixup_grad.py  --eval_ckpt experiments/test_l2_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma0.0025_lr0.002_nclass10_regularizerl2_repeat4_scales3_wd0.002/tf/model.ckpt-04587520 --train_dir experiments/test_l2_ema --dataset=cifar10.1@${size}-1 --nepoch 100 --regularizer l2 --gamma 0.0025 &
	CUDA_VISIBLE_DEVICES=3 python3 mixup_grad.py  --eval_ckpt experiments/test_l2_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma0.0025_lr0.002_nclass10_regularizerl2_repeat4_scales3_wd0.002/tf/model.ckpt-05242880 --train_dir experiments/test_l2_ema --dataset=cifar10.1@${size}-1 --nepoch 100 --regularizer l2 --gamma 0.0025 &
	CUDA_VISIBLE_DEVICES=3 python3 mixup_grad.py  --eval_ckpt experiments/test_l2_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma0.0025_lr0.002_nclass10_regularizerl2_repeat4_scales3_wd0.002/tf/model.ckpt-05898240 --train_dir experiments/test_l2_ema --dataset=cifar10.1@${size}-1 --nepoch 100 --regularizer l2 --gamma 0.0025 &
	CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py  --eval_ckpt experiments/test_l2_ema/cifar10.1@${size}-1/MixupGrad_LH1.0_archresnet_batch64_beta0.5_ema0.999_filters32_gamma0.0025_lr0.002_nclass10_regularizerl2_repeat4_scales3_wd0.002/tf/model.ckpt-06553600 --train_dir experiments/test_l2_ema --dataset=cifar10.1@${size}-1 --nepoch 100 --regularizer l2 --gamma 0.0025
done

# common_args="--regularizer None"
# # CIFAR 10
# for seed in 1 2 3 4 5; do
# 	for size in 100 500 1000 2500 5000 25000 50000; do
# 		CUDA_VISIBLE_DEVICES=1 python3 grad_regularization.py $common_args --eval_ckpt /data/dro-id_history/mixup10@${size}/cifar10.${seed}@${size}-1/FSMixup_archresnet_batch64_beta0.5_ema0.999_filters32_lr0.002_nclass10_repeat4_scales3_wd0.002 --train_dir /data/dro-id_history/mixup10@${size} --dataset=cifar10.${seed}@${size}-1
# 	done
# done

# # CIFAR 100
# for seed in 1 2 3 4 5; do
# 	for size in 1000 2500 5000 25000 50000; do
# 		CUDA_VISIBLE_DEVICES=2 python3 grad_regularization.py $common_args --eval_ckpt /data/dro-id_history/mixup100@${size}/cifar100.${seed}@${size}-1/FSMixup_archresnet_batch64_beta0.5_ema0.999_filters32_lr0.002_nclass100_repeat4_scales3_wd0.002 --train_dir /data/dro-id_history/mixup100@${size} --dataset=cifar100.${seed}@${size}-1
# 	done
# done
















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