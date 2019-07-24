
## Distributionally Robust Optimization with Interpolated Data

```bash
.
├── erm.py
├── input
│   ├── create_datasets.py
│   └── create_split.py
├── libml
│   ├── data.py
│   ├── __init__.py
│   ├── layers.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
├── mixup.py
├── README.md
├── requirements.txt
└── runs
    ├── cifar10-erm-mixup.yml
    └── prepare_datasets.sh
```

# Prepare datasets
```
sh runs/prepare_datasets.sh
```

# Example (CIFAR-10)
```
tmuxp load runs/cifar10-erm@10000.yml
tmuxp load runs/cifar10-erm@20000.yml
tmuxp load runs/cifar10-erm@30000.yml
tmuxp load runs/cifar10-erm@40000.yml
tmuxp load runs/cifar10-erm@50000.yml

tmuxp load runs/cifar10-mixup@10000.yml
tmuxp load runs/cifar10-mixup@20000.yml
tmuxp load runs/cifar10-mixup@30000.yml
tmuxp load runs/cifar10-mixup@40000.yml
tmuxp load runs/cifar10-mixup@50000.yml
```


