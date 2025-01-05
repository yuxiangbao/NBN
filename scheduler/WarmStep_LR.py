from torch.optim.lr_scheduler import _LRScheduler
import warnings

class WarmStepLR(_LRScheduler):
    '''refer to CMO'''

    def __init__(self, optimizer):
        super(WarmStepLR, self).__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch

        if epoch <= 5:
            return [lr * epoch/5 for lr in self.base_lrs]
        elif epoch > 80:
            return [lr * 0.01 for lr in self.base_lrs]
        elif epoch > 60:
            return [lr * 0.1 for lr in self.base_lrs]
        else:
            return self.base_lrs