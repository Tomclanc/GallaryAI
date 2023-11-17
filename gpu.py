import torch

# Check if a GPU is available and if PyTorch can see it
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
    print("Number of GPUs available:", torch.cuda.device_count())
    print("GPU device name:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("GPU is not available. The code will run on CPU.")
