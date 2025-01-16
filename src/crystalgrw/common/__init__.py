import torch
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)
print(f"Default tensor type: {DTYPE}")
