import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import pdb

class Linear_fcbn(nn.Linear):
    "The head medium tail share a common feature mask"
    def __init__(self, *args, num_tasks=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self, x, alpha_weights=None):
        x = F.linear(x, self.weight, self.bias)
        x = self.bn(x) 

        return x

class Linear_fcbn_notaffine(nn.Linear):
    "The head medium tail share a common feature mask"
    def __init__(self, *args, num_tasks=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.bn = nn.BatchNorm1d(self.out_features, affine=False)

    def forward(self, x, alpha_weights=None):   
        x = F.linear(x, self.weight, self.bias)
        x = self.bn(x) 
        return x
