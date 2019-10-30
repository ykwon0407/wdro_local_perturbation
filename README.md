
# Distributionally Robust Optimization with Interpolated Data
In this repository, we provide all python codes used in the paper 'Distributionally Robust Optimization with Interpolated Data'.

## Prepare datasets
The following simple code produces all the `tfrecord` files used in this repository.
```
sh runs/prepare_datasets.sh
```
The produced `tfrecord` files are named as `${dataset}.${seed}@${train_size}-{valid_size}` for each `seed` and `train_size`. For example, `cifar10.1@5000-1.tfrecord`

## Train models
Example 1. Train ERM model with `cifar10.1@50000-1.tfrecord` dataset.
```
CUDA_VISIBLE_DEVICES=0 python3 erm.py --dataset=cifar10.1@50000-1 --wd=0.02 --smoothing 0.001
```

<!---This code creates `./experiments/erm/cifar10.1@50000-1/tf, args` directory and save model checkpoints and arguments. In addition, this code also saves train, validation and test accuracy on each epochs at `./experiments/erm/cifar10.1@50000-1/accuracies.txt`--->

Example 2. Train mixup model with cifar100.5@5000-1 dataset by 100 epochs (default 128 epochs)
```
CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --dataset=cifar100.5@5000-1 --nepoch 100
```

<!---This code creates `./experiments/mixup/cifar100.1@5000-1/tf, args` directory and save model checkpoints and arguments. In addition, this code also saves train, validation and test accuracy on each epochs at `./experiments/mixup/cifar100.1@5000-1/accuracies.txt`--->

Example 3. Train l2 regularized model with cifar100.5@5000-1 dataset and gamma=0.1234
```
CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --dataset=cifar100.5@5000-1 --regularizer l2 --gamma 0.1234
```
This code creates `./experiments/l2_0.1234/cifar100.1@5000-1/tf, args` directory and save model checkpoints and arguments. In addition, this code also saves train, validation and test accuracy on each epochs at `./experiments/l2_0.1234/cifar100.1@5000-1/accuracies.txt`

## Evaluation with trained models
### Accuracy with noisy images
Example 1. Evaluation of the ERM model trained with 6553600 images.
```
CUDA_VISIBLE_DEVICES=1 python3 erm.py --eval_ckpt experiments/erm/cifar100.1@1000-1/tf/model.ckpt-06553600 -dataset=cifar100.1@1000-1 --noise_mode 'gaussian' --noise_var 0.01
```
This code evaluates accuracy using train, valid and test images affected by Gaussian noise with sigma=0.01, and save at `./experiments/erm/cifar100.1@1000-1/noise.txt`


Example 2. Evaluation of the mixup model trained with 6553600 images.
```
CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --eval_ckpt experiments/mixup/cifar10.4@1000-1/tf/model.ckpt-06553600 --dataset=cifar10.4@1000-1 --noise_mode 's&p' --noise_p 0.01
```
This code evaluates accuracy using train, valid and test images affected by salt and pepper noise with p=0.01, and save at `./experiments/mixup/cifar10.4@1000-1/noise.txt`


Example 3. Evaluation of the l2 (gamma=0.1234) model trained with 6553600 images.
```
CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py --eval_ckpt experiments/l2_0.1234/cifar100.1@1000-1/tf/model.ckpt-06553600 --dataset=cifar100.1@1000-1 --noise_mode 'Gaussian' --noise_var 0.005
```
This code evaluates accuracy using train, valid and test images affected by Gaussian noise with sigma=0.005, and save at `./experiments/l2_0.1234/cifar100.1@1000-1/noise.txt`


### Gradients of loss with respect to 10,000 test images.
Example 1. Evaluation with the ERM model saved at checkpoint: `experiments/erm/cifar10.1@100-1/tf/model.ckpt-06553600`
```
CUDA_VISIBLE_DEVICES=0 python3 erm.py  --eval_ckpt experiments/erm/cifar10.1@100-1/tf/model.ckpt-06553600 --dataset=cifar10.1@100-1
```
This code evaluates sup norm of each gradient vectors and save at `./experiments/erm/cifar10.1@100-1/gradients-06553600.txt`

Example 2. Evaluation with the mixup model saved at checkpoint: `experiments/mixup/cifar100.4@1000-1/tf/model.ckpt-01310720`
```
CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py  --eval_ckpt experiments/mixup/cifar100.4@1000-1/tf/model.ckpt-01310720 --dataset=cifar100.4@1000-1
```
This code evaluates sup norm of each gradient vectors and save at `./experiments/mixup/cifar100.4@1000-1/gradients-01310720.txt`

Example 3. Evaluation with the l2 (gamma=0.1234) model saved at checkpoint: `experiments/l2_0.1234/cifar10.5@50000-1/tf/model.ckpt-04587520`
```
CUDA_VISIBLE_DEVICES=0 python3 mixup_grad.py  --eval_ckpt experiments/l2_0.1234/cifar10.5@50000-1/tf/model.ckpt-04587520 --dataset=cifar10.5@50000-1
```
This code evaluates sup norm of each gradient vectors and save at `./experiments/l2_0.1234/cifar10.5@50000-1/gradients-04587520.txt`


## Directory tree

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
