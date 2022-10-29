import platform
from torch import cuda

def cuda_available():
    if cuda.is_available(): return 'cuda:0'
    else: return 'cpu'

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


if __name__ == "__main__":
    device_env()