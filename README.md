
## Distributionally Robust Optimization with Interpolated Data

```bash
.
├── erm.py
├── experiments
├── input
│   ├── create_datasets.py
│   └── create_split.py
├── libml
│   ├── data.py
│   ├── extract_accuracies_wy.py
│   ├── extract_accuracies_yc.py
│   ├── __init__.py
│   ├── layers.py
│   ├── models.py
│   ├── spectral_norm.py
│   ├── train.py
│   └── utils.py
├── mixup_grad.py
├── notebook
├── README.md
├── requirements.txt
└── runs
```

# Prepare datasets
```
sh runs/prepare_datasets.sh
```
This code creates tfrecord files for each seed and size formed as `${dataset}.${seed}@${train_size}-{valid_size}`. For example, `cifar10.1@5000-1.tfrecord`

# Train
Example 1. Train ERM model with cifar10.1@50000-1 dataset, weight decay = 0.02, and smoothing = 0.001
```
CUDA_VISIBLE_DEVICES=0 python3 erm.py --dataset=cifar10.1@50000-1 --wd=0.02 --smoothing 0.001
```
This code creates `./experiments/erm/cifar10.1@50000-1/tf, args` directory and save model checkpoints and arguments.

Example 2. Train mixup model with cifar100.5@5000-1 dataset by 100 epochs (default 128 epochs)
```
CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --dataset=cifar100.5@5000-1 --nepoch 100
```
This code creates `./experiments/mixup/cifar100.1@5000-1/tf, args` directory and save model checkpoints and arguments.

Example 3. Train l2 regularized model with cifar100.5@5000-1 dataset and gamma=0.1234
```
CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --dataset=cifar100.5@5000-1 --regularizer l2 --gamma 0.1234
```
This code creates `./experiments/l2_0.1234/cifar100.1@5000-1/tf, args` directory and save model checkpoints and arguments.

# Evaluation with trained models
## Accuracy with noisy images
Example 1. Evaluation of mixup model trained with 6553600 images.
```
CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --eval_ckpt experiments/mixup/cifar10.4@1000-1/tf/model.ckpt-06553600 -dataset=cifar10.4@1000-1 --noise_p 0.01 --regularizer='None' --noise_mode 's&p'
```
This code evaluates train, valid and test images with salt and pepper noise with p=0.01.

Example 2. Evaluation of ERM model trained with 6553600 images.
```
CUDA_VISIBLE_DEVICES=1 python3 erm.py --eval_ckpt experiments/erm/cifar100.1@1000-1/tf/model.ckpt-06553600 -dataset=cifar100.1@1000-1 --noise_var 0.01 --noise_mode 'gaussian'
```
This code evaluates train, valid and test images with gaussian noise with sigma=0.01.

## Gradients of loss with respect to test images.
TBA
