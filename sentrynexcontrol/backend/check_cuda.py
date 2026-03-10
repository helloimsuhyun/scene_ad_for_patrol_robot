
import torch
import os
import subprocess

def check_basic():
    print("===== Basic Info =====")
    print("PyTorch version :", torch.__version__)
    print("CUDA version (torch) :", torch.version.cuda)
    print("CUDA_VISIBLE_DEVICES :", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("CUDA available :", torch.cuda.is_available())
    print("GPU count :", torch.cuda.device_count())
    print()

def check_gpu_info():
    print("===== GPU Info =====")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} :", torch.cuda.get_device_name(i))
            prop = torch.cuda.get_device_properties(i)
            print("  total memory (GB):", round(prop.total_memory / 1e9, 2))
            print("  compute capability:", f"{prop.major}.{prop.minor}")
    else:
        print("CUDA not available")
    print()

def check_tensor_compute():
    print("===== Tensor Compute Test =====")
    if not torch.cuda.is_available():
        print("Skip (CUDA not available)")
        return

    device = torch.device("cuda:0")
    print("Using device:", device)

    try:
        a = torch.randn((4096, 4096), device=device)
        b = torch.randn((4096, 4096), device=device)

        c = torch.matmul(a, b)

        print("Tensor matmul success")
        print("Result tensor device:", c.device)
        print("Result shape:", c.shape)

        mem_alloc = torch.cuda.memory_allocated(device) / 1e6
        print("GPU memory allocated (MB):", round(mem_alloc, 2))

    except Exception as e:
        print("Tensor compute failed:", repr(e))

    print()

def check_nvidia_smi():
    print("===== nvidia-smi =====")
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
    except Exception as e:
        print("nvidia-smi failed:", repr(e))

def main():
    check_basic()
    check_gpu_info()
    check_tensor_compute()
    check_nvidia_smi()

if __name__ == "__main__":
    main()