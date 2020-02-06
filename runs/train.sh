
# Train ERM with weight decay 0.02 and smoothing 0.001 for 100 ckpts.
for seed in 1 2 3 4 5; do
  for size in 2500 5000 25000 50000; do
    CUDA_VISIBLE_DEVICES= python3 erm.py --dataset=cifar10.$seed@$size-1 --wd=0.02 --smoothing 0.001 --nckpt 100
    CUDA_VISIBLE_DEVICES= python3 erm.py --dataset=cifar100.$seed@$size-1 --wd=0.02 --smoothing 0.001 --nckpt 100
  done
done

# Train MIXUP for 100 ckpts.
for seed in 1 2 3 4 5; do
  for size in 2500 5000 25000 50000; do
    CUDA_VISIBLE_DEVICES= python3 mixup_grad.py --dataset=cifar10.$seed@$size-1 --nckpt 100
    CUDA_VISIBLE_DEVICES= python3 mixup_grad.py --dataset=cifar100.$seed@$size-1 --nckpt 100
  done
done

# Train WDRO for 100 ckpts.
for seed in 1 2 3 4 5; do
  for size in 2500 5000 25000 50000; do
    CUDA_VISIBLE_DEVICES= python3 erm.py --dataset=cifar10.$seed@$size-1 --gamma 0.004 --nckpt 100
    CUDA_VISIBLE_DEVICES= python3 erm.py --dataset=cifar100.$seed@$size-1 --gamma 0.004 --nckpt 100
  done
done

# Train WDRO+MIX for 100 ckpts.
for seed in 1 2 3 4 5; do
  for size in 2500 5000 25000 50000; do
    CUDA_VISIBLE_DEVICES= python3 mixup_grad.py --dataset=cifar10.$seed@$size-1 --gamma 0.004 --nckpt 100
    CUDA_VISIBLE_DEVICES= python3 mixup_grad.py --dataset=cifar100.$seed@$size-1 --gamma 0.004 --nckpt 100
  done
done
