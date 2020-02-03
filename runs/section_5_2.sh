# Evaluation of the gradients of a loss for Section 5.2
for steps in 00655360 01310720 01966080 02621440 03276800 03932160 04587520 05242880 05898240 06553600; do
  for seed in 1 2 3 4 5; do
    for size in 2500 5000 25000 50000; do
      for dataset in cifar10 cifar100; do
        CUDA_VISIBLE_DEVICES= python3 erm.py --eval_ckpt experiments/ERM/$dataset.$seed@$size-1/tf/model.ckpt-$steps -dataset=$dataset.$seed@$size-1
        CUDA_VISIBLE_DEVICES= python3 erm.py --eval_ckpt experiments/WDRO_0.004/$dataset.$seed@$size-1/tf/model.ckpt-$steps -dataset=$dataset.$seed@$size-1
        CUDA_VISIBLE_DEVICES= python3 mixup_grad.py --eval_ckpt experiments/MIXUP/$dataset.$seed@$size-1/tf/model.ckpt-$steps -dataset=$dataset.$seed@$size-1
        CUDA_VISIBLE_DEVICES= python3 mixup_grad.py --eval_ckpt experiments/WDRO_MIX_0.004/$dataset.$seed@$size-1/tf/model.ckpt-$steps -dataset=$dataset.$seed@$size-1
			done
		done
	done
done
