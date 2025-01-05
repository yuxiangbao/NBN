import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import pdb

class RIDE_BalancedSoftmaxLoss(nn.Module):
    def __init__(self, freq, base_diversity_temperature=1.0,
        additional_diversity_factor=-0.2, reduction="mean", **kwargs):
        super().__init__()
        self.base_loss = F.cross_entropy
        freq = torch.tensor(freq)
        self.sample_per_class = freq
        self.reduction = reduction

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def _hook_before_epoch(self, epoch):
        pass
    
    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0

        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:
            ride_loss_logits = logits_item
            loss += balanced_softmax_loss(target, ride_loss_logits, self.sample_per_class, self.reduction)
            
            if self.additional_diversity_factor:
                base_diversity_temperature = self.base_diversity_temperature
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature            
                output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
                with torch.no_grad():
                    # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                    mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)
                
                loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist, reduction='batchmean')
        
        return loss


def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss