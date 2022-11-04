import argparse

import torch.optim as optim
import torch.nn as nn

from torchmetrics.functional.classification import accuracy
from src.engines import DS

from src import utils
from src import datasets
from src.models import DSNet


def main():
    parser = argparse.ArgumentParser(description="dsnet with cifar-10")
    utils.arg_setting(parser)
    parser.add_argument("--title", default="dsnet-cifar10", type=str)
    args = parser.parse_args()
    
    utils.device_env()
    
    # base setting
    device = utils.device_setting(args.device)
    print(f"Torch running on {device}")
    
    # Load dataset with dataloader
    cifar_dataload = datasets.CIFAR10.load_train_cifar_10(root=args.data, batch_size=4, shuffle=True, num_workers=args.numworkers)

    
    # Build Model
    model = DSNet(device, "checkpoints/da-cifar10.pth")
    model = model.to(device)
    
    
    # Build optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(cifar_dataload))
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = accuracy
    
    # Loop
    for epoch in range(args.epochs):
        summary = DS.train(cifar_dataload, model, optimizer, scheduler, loss_fn, metric_fn, device)
        print(f'Epoch: {epoch + 1}, Accuracy: {summary["metric"]:.4f}')
        utils.save_checkpoint(args.checkpoints, args.title, model, optimizer, epoch + 1)

if __name__ == "__main__":
    main()
    