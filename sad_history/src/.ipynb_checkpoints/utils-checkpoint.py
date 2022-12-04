import platform
from torch import cuda
import argparse
import torch
import os

def cuda_available():
    if cuda.is_available():
        if cuda.device_count() != 1: return 'cuda:0'
        else: return 'cuda'
    else: return 'cpu'

def device_setting(arg):
    if arg == "auto":
        return cuda_available()
    else:
        return arg

def device_env():

    print(f"Platform : {platform.system()}")

    try:
        from cpuinfo import get_cpu_info

        info = get_cpu_info()

        print(f"Processor : {info['brand_raw']} - {info['arch_string_raw']}")
    except ImportError:
        print("Can't import cpuinfo module, please install py-cpuinfo - 'pip install py-cpuinfo'")

    if cuda.is_available():
        print("CUDA is available")
        for i in range(cuda.device_count()):
            print(f"{cuda.get_device_name(i)}")
    else:
        print("CUDA is unavailable")

def arg_setting(parser: argparse.ArgumentParser):
    import src.global_args_setting as GAS
    parser.add_argument("--lr", default=GAS.LEARNING_RATE, type=float, help="learning rate")
    parser.add_argument("--epochs", default=GAS.EPOCHS, type=int, help="epoch")
    parser.add_argument("--checkpoints", default=GAS.CHECKPOINTS, type=str, help="checkpoints dir")
    parser.add_argument("--numworkers", default=GAS.NUM_WORKERS, type=int, help="num workers")
    parser.add_argument("--data", default=GAS.DATA, type=str, help="download and load data dir")
    parser.add_argument("--device", default=GAS.DEVICE, type=str, help="cpu or cuda if available")

def save_checkpoint(checkpoint_dir, title, model, optimizer, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_dict = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    checkpoint_path = f'{checkpoint_dir}/{title}.pth'
    torch.save(state_dict, checkpoint_path)

if __name__ == "__main__":
    device_env()