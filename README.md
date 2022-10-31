# ml2-project


This is a repository about the project of ML2 class, SEOULTECH😄.


# Environment Setting

### Install
Install conda

For A5000
```
conda create -n "ml2" python=3.9
conda activate "ml2"
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
```

Set VSC default python interpreter as "ml2" python 3.9.10

# Run

### Move to ml2 dir
```
cd ml2
```

### TestNet-CIFAR10
```
python testnet-cifar10.py
```

### TestNet-CIFAR10 with data augmentation
```
python testnet-cifar10.py --title aug-testnet-cifar10
```

### ResNet18-CIFAR10
```
python resnet-cifar10.py
```

# Evaluate

### Move to ml2 dir
```
cd ml2
```

### TestNet-CIFAR10
```
python eval-cifar10.py --title testnet-cifar10
```

### TestNet-CIFAR10 with data augmentation
```
python eval-cifar10.py --title aug-testnet-cifar10
```