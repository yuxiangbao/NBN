from torch.optim.lr_scheduler import _LRScheduler
import warnings

class CIFARWarmStepLR(_LRScheduler):
    '''refer to CMO'''

    def __init__(self, optimizer):
        super(CIFARWarmStepLR, self).__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch

        if epoch <= 5:
            return [lr * epoch/5 for lr in self.base_lrs]
        elif epoch > 180:
            return [lr * 0.001 for lr in self.base_lrs]
        elif epoch > 160:
            return [lr * 0.01 for lr in self.base_lrs]
        else:
            return self.base_lrs