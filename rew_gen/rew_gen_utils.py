import torch
import os
import numpy as np

def tensor(x,device):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(device)
    return x
