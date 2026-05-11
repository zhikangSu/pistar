import torch

def safe_stack(tensors, *args, **kwargs):
    print("Safe stack called!")
    return torch.tensor([1, 2, 3])

original_stack = torch.stack
torch.stack = safe_stack

res = torch.stack([torch.tensor(1)])
print("Result:", res)

torch.stack = original_stack
