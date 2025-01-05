import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import pdb

class HCMLoss(_Loss):
    """
    Hard Category Mining Loss
    """
    def __init__(self, freq, HCM_N=300, reduction='mean', **kwargs):
        super(HCMLoss, self).__init__()
        freq = torch.tensor(freq)
        self.sample_per_class = freq
        self.hcm_N = HCM_N
        self.reduction = reduction

    def forward(self, input, label):
        loss = 0
        loss = loss + balanced_softmax_loss(label, input, self.sample_per_class, self.reduction)
        
        class_select = input.scatter(1, label.unsqueeze(1), 999999)
        class_select_include_target = class_select.sort(descending=True, dim=1)[1][:, :self.hcm_N] 
        mask = torch.zeros_like(input).scatter(1, class_select_include_target, 1)
        loss = loss + balanced_softmax_loss(label, input, self.sample_per_class, self.reduction, mask=mask)
        return loss


def balanced_softmax_loss(labels, logits, sample_per_class, reduction, mask=None):
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
    if mask is not None:
      logits = logits * mask
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss
