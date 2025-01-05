import torch
import torch.nn as nn
import numpy as np

import pdb

def plain_train(model, x_natural, y, criterion, **kwargs):
    logits = model(x_natural)
    loss = criterion(logits, y)
    return logits, loss

def plain_train_ensemble(model, x_natural, y, criterion, epoch, **kwargs):
    output = model(x_natural)
    logits_mean = output["output"]
    extra_info = dict(logits=output["logits"].transpose(0,1))
    
    criterion._hook_before_epoch(epoch)
    loss = criterion(logits_mean, y, extra_info=extra_info)
    return logits_mean, loss


