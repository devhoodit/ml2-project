import argparse
from multiprocessing.sharedctypes import Value
from turtle import title

import torch
from torchmetrics.functional.classification import accuracy
import torch.nn as nn

from src.models import TestNet
from src.engines import Test
from src import utils, datasets

def main():
    
    parser = argparse.ArgumentParser(description="resnet18 with cifar-10")
    utils.arg_setting(parser)
    parser.add_argument("--title", default="testnet-cifar10", type=str)
    args = parser.parse_args()
    
    utils.device_env()
    
    # base setting
    device = utils.device_setting(args.device)
    print(f"Torch running on {device}")
    
    # Dataload
    cifar_dataload = datasets.CIFAR10.load_test_cifar_10(root=args.data, batch_size=4, shuffle=True, num_workers=args.numworkers)
    
    # Load Model
    if model_name := args.title == 'testnet-cifar10' or 'aug-testnet-cifar10':
        from src.models import TestNet
        model = TestNet()
    elif model_name == 'resnet18-cifar10' or 'aug-resnet18-cifar10':
        from src.models import ResNet18
        model = ResNet18()
    else:
        raise ValueError(f'{model_name} is not exist')
    
    state_dict = torch.load(f'{args.checkpoints}/{args.title}.pth')
    model.load_state_dict(state_dict['model'])
    model = model.to(device)
    
    # Build loss and metric
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = accuracy
    
    # Evaluate
    summary = Test.evaluate(cifar_dataload, model, loss_fn, metric_fn, device)
    acc = summary['metric']

    print(f'Model: {args.title} Accuracy: {acc}')
    
    return acc

if __name__ == "__main__":
    main()
    
    