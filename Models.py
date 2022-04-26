from numpy import size
import torch
import torch.nn as nn

def activation_functions(activation):
    
    dict = { "relu":nn.ReLU(),"leakyrelu":nn.LeakyReLU(),"sigmoid":nn.Sigmoid(),"tanh":nn.Tanh()}
    assert activation in dict, "unknown activation function"
    return dict[activation]

class Linear(nn.Module):
    
    def __init__(self,sizes,activation=None,final_activation=None):
        super().__init__()
        layers = []
        for i in range(len(sizes)-2):
            layers.append(nn.Linear(sizes[i],sizes[i+1]))
            if activation:
                layers.append(activation_functions(activation))
        layers.append(nn.Linear(sizes[-2],sizes[-1]))
        if final_activation:
            layers.append(activation_functions(final_activation))
        self.sequential = nn.Sequential(*layers)
    
    def forward(self,x):
        return self.sequential(x)
