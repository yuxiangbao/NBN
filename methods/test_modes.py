import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from .utils import entropy, target2task, get_adv_prob_per_sample, save_adv
from math import ceil


def forward(model, x_natural, y, criterion, **kwargs):
    logits = model(x_natural)
    loss = criterion(logits, y)
    return logits, loss

def forward_ensemble(model, x_natural, y, criterion, **kwargs):
    logits = model(x_natural)["output"]
    loss = criterion(logits, y)
    return logits, loss