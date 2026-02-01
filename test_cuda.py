import torch

print("=== CUDA 기본 정보 ===")
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    print("Current device id:", device_id)
    print("Device name:", torch.cuda.get_device_name(device_id))

    print("\n=== GPU 연산 테스트 ===")
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()

    z = torch.matmul(x, y)
    print("Matrix multiplication success.")
    print("Result device:", z.device)

else:
    print("CUDA not available.")
