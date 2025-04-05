import torch
import torch_directml

# Check if DirectML is available
dml_available = torch_directml.is_available()
print(f"DirectML available: {dml_available}")

if dml_available:
    # Get the DirectML device
    dml_device = torch_directml.device()
    print(f"DirectML device: {dml_device}")

    # You can also get the device name (though it might not be as descriptive as NVIDIA/CUDA)
    print(f"Device name: {torch.device(dml_device).type}") # Will likely output 'dml'

    # Example of moving a tensor to the DirectML device
    tensor = torch.randn(5, 3)
    tensor_on_dml = tensor.to(dml_device)
    print(f"Tensor on DirectML device: {tensor_on_dml.device}")

    # Example of creating a tensor directly on the DirectML device
    tensor_direct = torch.ones(2, 2, device=dml_device)
    print(f"Tensor created directly on DirectML: {tensor_direct.device}")
else:
    print("DirectML is not available. Ensure you have installed 'torch-directml' correctly.")

# For comparison, let's also check for CUDA availability (though it won't be relevant for your AMD/DirectML setup)
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    cuda_device = torch.device("cuda")
    print(f"CUDA device: {cuda_device}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device index: {torch.cuda.current_device()}")
    print(f"CUDA device name (index 0): {torch.cuda.get_device_name(0)}")