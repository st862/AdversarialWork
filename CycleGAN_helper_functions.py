import torch
import torch.optim as optim
import torch.nn as nn

def LOSS_D(real,fake):
    return torch.mean((real-1)**2) + torch.mean(fake**2)

def LOSS_G(fake):
    return torch.mean((fake-1)**2)

def Epoch_G():
    pass

def Epoch_D():
    pass


def Epoch_AE():
    pass

def Save_Models():
    pass

def Load_Models():
    pass