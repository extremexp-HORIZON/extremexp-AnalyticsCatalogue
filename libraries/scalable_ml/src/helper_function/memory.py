import torch
from torch import nn

def measure_step_memory(model, example_input, device="cuda", do_backward=True):



    model = model.to(device)
    example_input = example_input.to(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # Optional: ensure things are synced before measurement
    torch.cuda.synchronize(device)

    # Record memory before
    mem_before = torch.cuda.memory_allocated(device)

    # Forward
    output = model(example_input)
    loss = output.sum()

    # Backward (optional)
    if do_backward:
        loss.backward()

    # Ensure all kernels done
    torch.cuda.synchronize(device)

    mem_after = torch.cuda.memory_allocated(device)
    mem_peak = torch.cuda.max_memory_allocated(device)

    print(f"Memory before: {to_mb(mem_before):.2f} MB")
    print(f"Memory after:  {to_mb(mem_after):.2f} MB")
    print(f"Peak during:   {to_mb(mem_peak):.2f} MB")

    return mem_before, mem_after, mem_peak

def print_cuda_memory(device="cuda"):
    allocated = torch.cuda.memory_allocated(device)
    reserved  = torch.cuda.memory_reserved(device)

    print(f"Allocated: {to_mb(allocated):.2f} MB")
    print(f"Reserved:  {to_mb(reserved):.2f} MB")


def to_mb(bytes_):
    return bytes_ / (1024 ** 2)

def get_model_param_size(model: torch.nn.Module):
    """
    Get the footprint of the model using the parameters
    # return the total in bytes
    # input a torch model
    """

    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    total_size = param_size + buffer_size

    print(f"Parameters: {to_mb(param_size):.2f} MB")
    print(f"Buffers:    {to_mb(buffer_size):.2f} MB")
    print(f"Total:      {to_mb(total_size):.2f} MB")

    return total_size  # in bytes    

