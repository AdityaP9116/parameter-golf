import torch
import math

def fwht(x):
    d = x.shape[-1]
    h = 1
    while h < d:
        x_reshaped = x.view(*x.shape[:-1], d // (h * 2), 2, h)
        x0 = x_reshaped[..., 0, :]
        x1 = x_reshaped[..., 1, :]
        x = torch.stack([x0 + x1, x0 - x1], dim=-2).view(*x.shape[:-1], d)
        h *= 2
    return x * (d ** -0.5)

x = torch.eye(8)
print(fwht(x))
