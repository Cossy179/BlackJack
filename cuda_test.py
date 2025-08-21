import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))

# Example tensor
x = torch.rand(1000, 1000).to(device)
print("Tensor device:", x.device)

# Test a simple operation
y = torch.rand(1000, 1000).to(device)
z = torch.matmul(x, y)
print("Matrix multiplication successful on:", z.device)
