import argparse

import torch
from torchmetrics.functional.classification import accuracy
import torch.nn as nn

from src import utils, datasets

def main():
    
    parser = argparse.ArgumentParser(description="resnet18 with cifar-10")
    utils.arg_setting(parser)
    parser.add_argument("--title", default="testnet-cifar10", type=str)
    args = parser.parse_args()
    
    utils.device_env()

    model_name = args.title
    
    # base setting
    device = utils.device_setting(args.device)
    print(f"Torch running on {device}")
    print(f"Model: {model_name}")
    
    # Dataload
    if model_name == "da-cifar10":
        cifar_dataload = datasets.CIFAR10DataAug.load_test(root=args.data)
    else:
        cifar_dataload = datasets.CIFAR10.load_test_cifar_10(root=args.data, batch_size=4, shuffle=True, num_workers=args.numworkers)
    
    # Load Model
    if model_name == 'testnet-cifar10' or model_name == 'aug-testnet-cifar10':
        from src.models import TestNet
        model = TestNet()
    elif model_name == 'resnet18-cifar10' or model_name == 'aug-resnet18-cifar10':
        from src.models import ResNet18
        model = ResNet18()
    elif model_name == 'da-cifar10':
        from src.models import DataAugNet
        model = DataAugNet()
    else:
        raise ValueError(f'{model_name} is not exist')
    
    state_dict = torch.load(f'{args.checkpoints}/{args.title}.pth')
    model.load_state_dict(state_dict['model'])
    model = model.to(device)
    
    # Build loss and metric
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = accuracy
    
    # Evaluate
    if model_name := args.title == 'testnet-cifar10' or model_name == 'aug-testnet-cifar10':
        from src.engines import Test
        summary = Test.evaluate(cifar_dataload, model, loss_fn, metric_fn, device)
    elif model_name == 'resnet18-cifar10' or model_name == 'aug-resnet18-cifar10':
        from src.engines import ResNet18CIFAR10
        summary = ResNet18CIFAR10.evaluate(cifar_dataload, model, loss_fn, metric_fn, device)
    elif model_name == 'da-cifar10':
        from src.engines import DACIFAR10
        summary = DACIFAR10.evaluate(cifar_dataload, model, loss_fn, metric_fn, device)
    acc = summary['metric']

    print(f'Model: {args.title} Accuracy: {acc}')
    
    return acc

if __name__ == "__main__":
    main()
    
    