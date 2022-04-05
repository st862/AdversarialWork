import torch
import torch.optim as optim
import torch.nn as nn

def LOSS_D(real,fake):
    return torch.mean((real-1)**2) + torch.mean(fake**2)

def LOSS_G(fake):
    return torch.mean((fake-1)**2)

