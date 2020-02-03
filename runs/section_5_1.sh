# Evaluate accuracy on clean and noisy datasets for Section 5.1
for seed in 1 2 3 4 5; do
  for size in 2500 5000 25000 50000; do
    for dataset in cifar10 cifar100; do
      CUDA_VISIBLE_DEVICES= python3 erm.py --eval_ckpt experiments/ERM/$dataset.$seed@$size-1/tf/model.ckpt-06553600 -dataset=$dataset.$seed@$size-1 --noise_p 0.01
      CUDA_VISIBLE_DEVICES= python3 erm.py --eval_ckpt experiments/WDRO_0.004/$dataset.$seed@$size-1/tf/model.ckpt-06553600 -dataset=$dataset.$seed@$size-1 --noise_p 0.01
      CUDA_VISIBLE_DEVICES= python3 mixup_grad.py --eval_ckpt experiments/MIXUP/$dataset.$seed@$size-1/tf/model.ckpt-06553600 -dataset=$dataset.$seed@$size-1 --noise_p 0.01
      CUDA_VISIBLE_DEVICES= python3 mixup_grad.py --eval_ckpt experiments/WDRO_MIX_0.004/$dataset.$seed@$size-1/tf/model.ckpt-06553600 -dataset=$dataset.$seed@$size-1 --noise_p 0.01
    done
  done
done
