from enum import Enum
import torch
import torch.distributed as dist
import numpy as np

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class StatMeter(object):
    def __init__(self, task_strs, task_content, num_classes):
        self.task_strs = task_strs
        self.task_content = task_content
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        num_tasks = len(self.task_strs)
        self.num = torch.zeros(num_tasks).cuda()
        self.mean = torch.zeros(num_tasks).cuda()
        self.var = torch.zeros(num_tasks).cuda()
        self.minmax = torch.zeros((num_tasks,2)).cuda()
        self.minmax[:, 0] = 666

    def update(self, output, task_id):
        num = len(output)
        entropy = -(output.softmax(dim=1) * output.log_softmax(dim=1)).sum(1)               
        
        self.var[task_id] = (self.num[task_id]*self.var[task_id] + num*entropy.var(unbiased=False))/(self.num[task_id]+num) + \
                self.num[task_id]*num*torch.pow(self.mean[task_id]-entropy.mean(), 2)/torch.pow(self.num[task_id]+num, 2)                 
        self.mean[task_id] = (self.mean[task_id]*self.num[task_id] + entropy.sum()) / (self.num[task_id] + len(output))
        self.num[task_id] += len(output) 
        self.minmax[task_id, 0] = torch.min(self.minmax[task_id, 0], entropy.min())
        self.minmax[task_id, 1] = torch.max(self.minmax[task_id, 1], entropy.max())

class AttackMeter():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.num = torch.zeros(self.num_classes).cuda().double()
        self.ratio = torch.zeros(self.num_classes).cuda().double()

    def update(self, attack_target):
        uniq, counts = torch.unique(attack_target, return_counts=True)
        self.num[uniq] += counts
        self.ratio = self.num / self.num.sum()

    def all_reduce(self):
        dist.all_reduce(self.num, dist.ReduceOp.SUM, async_op=False)
        self.ratio = self.num / self.num.sum()