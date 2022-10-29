import argparse

import torch.optim as optim
import torch.nn as nn

from torchmetrics.functional.classification import accuracy
from ml2.src.datasets import CIFAR10
from src.engines import Test

from src import utils
from src import datasets
from src.models import TestNet

def main():
    parser = argparse.ArgumentParser(description="resnet18 with cifar-10")
    utils.arg_setting(parser)
    parser.add_argument("--title", default="testnet-cifar10", type=str)
    args = parser.parse_args()
    
    utils.device_env()
    
    # base setting
    device = utils.device_setting(args.device)
    print(f"Torch running on {device}")
    
    # Data augmentation
    if args.title == "aug-testnet-cifar10":
        from torchvision import transforms as T
        CIFAR10.transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        T.RandomErasing(),
    ])
    
    # Load dataset with dataloader
    cifar_dataload = datasets.CIFAR10.load_train_cifar_10(root=args.data, batch_size=4, shuffle=True, num_workers=args.numworkers)

    
    # Build Model
    model = TestNet()
    model = model.to(device)
    
    
    # Build optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(cifar_dataload))
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = accuracy
    
    # Loop
    print("Running")
    for epoch in range(args.epochs):
        summary = Test.train(cifar_dataload, model, optimizer, scheduler, loss_fn, metric_fn, device)
        print(f'Epoch: {epoch + 1}, Accuracy: {summary["metric"]:.4f}')
        utils.save_checkpoint(args.checkpoints, args.title, model, optimizer, epoch + 1)
    
if __name__ == "__main__":
    main()