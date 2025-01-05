import argparse
import os
import random
import yaml
import shutil
import copy
import time
import warnings
from data import dataloader
import loss
import scheduler as lr_scheduler
from meter import AverageMeter, ProgressMeter, Summary, AttackMeter
from methods.utils import (freeze_model, load_pickle, get_task_content, 
                target2task,train_fc_only, update, save_plot_atk_tgts)
from logger import create_log_dir, get_root_logger, logger_info, copy_config
from methods import test_modes, train_modes

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.cuda.amp as amp
# from tensorboardX import SummaryWriter
import numpy as np
import pdb
import matplotlib.pyplot as plt
import models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
task_strs = ["low", "median", "many"]
num_classes = 1000



parser = argparse.ArgumentParser(description='PyTorch Long-Tail Training')

parser.add_argument('cfg', default=None, type=str)
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='use pre-trained model')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:6666', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('-d', '--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-g', '--task-given', action='store_false',
                    help='specify task given or not')
parser.add_argument('-w', '--weight_decay', action='store_false',
                    help='whether to implement weight decay on bias and scale factor')
parser.add_argument('--amp', action='store_true',
                    help='whether to implement auto mixed precision')
                    
best_acc1 = 0

def main():    
    args = parser.parse_args()

    # LOAD CONFIGURATIONS
    with open(args.cfg) as f:
        config = yaml.safe_load(f)
    args = update(config, args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if not args.evaluate:
        create_log_dir(args)
        copy_config(config, args)


    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, num_classes
    num_classes = args.num_classes
    args.gpu = gpu

    # logger = get_root_logger(log_dir=args.log_dir)

    # preparing model code
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=num_classes, **getattr(args, "arch_args", {}))
    model = model if args.train_all else train_fc_only(model, **getattr(args, "tune_fc_args", {}))
    if args.pretrained:
        print("=> using pre-trained backbone of model '{}'".format(args.arch))
        if args.gpu is None:
                checkpoint = torch.load(args.pretrained)["state_dict"]
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.pretrained, map_location=loc)["state_dict"]

        stat_dict = model.state_dict()        
        for k, v in checkpoint.items():
            k = k.replace("module.", "")
            if k in stat_dict.keys() and \
                (v.shape==stat_dict[k].shape):
                stat_dict.update({k: v})
        
        model.load_state_dict(stat_dict)

        if getattr(args, "rand_init_fc", False):
            for n, m in model.fc.named_modules():
                m.reset_parameters()
        if hasattr(model, "re_init_scale_factor"):
            model.re_init_scale_factor()
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # Data loading code
    dataset = args.dataset
    splits = ['train', 'train_plain', 'val']
    if dataset not in ['iNaturalist18', 'ImageNet']:
        splits.append('test')
    data = {x: dataloader.load_data(data_root=args.data_root,
                                    dataset=dataset, phase=x, 
                                    batch_size=args.batch_size,
                                    sampler_dic=None,
                                    shuffle=args.shuffle,
                                    num_workers=args.workers,
                                    distributed=args.distributed,
                                    resample=getattr(args, "resample", False),
                                    **args.data_args if hasattr(args, 'data_args') else {})
            for x in splits}
    task_content, num_classes, freq = get_task_content(data["train"])

    # define loss function (criterion) and optimizer
    criterion = build_criterion(args, freq=freq, cls_num_list=freq*len(data["train"].dataset))

    # define optimizer
    optimizer_cfg = args.optimizer
    optimizer_cls = getattr(torch.optim, optimizer_cfg.pop("type"))
    if args.weight_decay:
        optimizer = optimizer_cls(model.parameters(), **optimizer_cfg)
    else:
        params_s = (p for n, p in model.named_parameters() if "scale_factor" in n)
        params = (p for n, p in model.named_parameters() if "scale_factor" not in n)
        optimizer = optimizer_cls([{'params': params_s, 'weight_decay': 0}, 
                                    {'params': params, 'weight_decay': optimizer_cfg.pop("weight_decay")}], **optimizer_cfg)

    #define lr scheduler
    scheduler_cfg = args.scheduler
    scheduler_cls = getattr(lr_scheduler, 
                            scheduler_cfg.pop("type"))
    scheduler = scheduler_cls(optimizer, **scheduler_cfg)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']

            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = torch.tensor(best_acc1).to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # optionally load static model to make adv augmentation
    if args.train_mode.get("static_model", None):
        assert isinstance(args.train_mode["static_model"], str)
        static_model_path = args.train_mode["static_model"]
        static_model = copy.deepcopy(model)
        if os.path.isfile(static_model_path):
            if args.gpu is None:
                checkpoint = torch.load(static_model_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(static_model_path, map_location=loc)
            
            stat_dict = checkpoint['state_dict']
            if not args.distributed:
                stat_dict = {k.replace("module.", ""):v for k, v in stat_dict.items()}
        
            static_model.load_state_dict(stat_dict)
            args.train_mode["static_model"] = freeze_model(static_model) 

            print("=> loaded static model checkpoint '{}'"
                  .format(static_model_path))
        else:
            print("=> no checkpoint found at '{}'".format(static_model_path))

    cudnn.benchmark = True

    # evaluate
    if args.evaluate:
        validate(data["test"] if "test" in data else data["val"], model, criterion, task_content, args)
        return

    # train
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data["train"].sampler.set_epoch(epoch)
            
        # train for one epoch
        train(data["train"], model, criterion, optimizer, epoch, task_content, freq, args)

        # evaluate on validation set
        acc1 = validate(data["val"], model, criterion, task_content, args, epoch=epoch)

        # update lr
        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, args)
    eval_best_model(model, data, criterion, task_content, args)

def train(train_loader, model, criterion, optimizer, epoch, task_content, freq, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    if ("plain_train" not in args.train_mode.get("type", "plain_train") and 
        epoch >= args.train_mode.get("warm_up", 0) and 
        epoch % args.train_mode.get("atk_tgt_stat_epoch", 10)==0):
        attack_tgts = AttackMeter(args.num_classes)
        tgts = AttackMeter(args.num_classes)
    else:
        attack_tgts = None
        tgts = None
    # switch to train mode
    model.train()
    # mixed precision training
    scaler = amp.GradScaler()

    end = time.time()
    for i, (images, target, indexes) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)        

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)      

        # compute output & loss
        loss_params = dict(
            model=model, x_natural=images, y=target, optimizer=optimizer,
            criterion=criterion, task_content=task_content, task_strs=task_strs,
            num_classes=num_classes, freq=freq, epoch=epoch, iter=i, attack_tgts=attack_tgts, tgts=tgts)
        loss_params.update(args.train_mode)  

        with amp.autocast(enabled=args.amp): 
            output, loss = getattr(train_modes, 
                        args.train_mode.get("type"))(**loss_params)

        # measure accuracy and record loss
        acc1, acc5, n = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], n)
        top5.update(acc5[0], n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()        
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    save_plot_atk_tgts(attack_tgts, tgts, freq, epoch, args)

def validate(val_loader, model, criterion, task_content, args, epoch=None):
    # initialize meters
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    acc_percls = [AverageMeter('Acc_t{}@1'.format(i), ':6.2f', Summary.AVERAGE) for i in range(len(task_strs))]
    task_acc = [AverageMeter('TaskAcc_t{}@1'.format(i), ':6.2f', Summary.AVERAGE) for i in range(len(task_strs))]
    if args.task_given:
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5]+acc_percls,
            prefix='Test: ')
    else:
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5]+acc_percls+task_acc,
            prefix='Test: ')

    
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target, indexes) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            task = target2task(target, task_content, task_strs, num_classes)

            # compute output and record loss
            loss_params = dict(
            model=model, x_natural=images, y=target, criterion=criterion,
            task_content=task_content, task_strs=task_strs,num_classes=num_classes) 
            loss_params.update(getattr(args, "test_mode", {}))  

            test_mode = args.test_mode.get("type") if hasattr(args, "test_mode") else "forward"
            with amp.autocast(enabled=args.amp): 
                output, loss = getattr(test_modes, test_mode)(**loss_params)
            
            losses.update(loss.item(), output.size(0)) 

            for task_id in range(len(task_strs)):
                otpt, tgts = output[task==task_id], target[task==task_id]
                
                if otpt.size(0):                
                    # measure accuracy
                    acc1, acc5, n = accuracy(otpt, tgts, topk=(1, 5))    
                    top1.update(acc1[0], n)
                    top5.update(acc5[0], n)
                    acc_percls[task_id].update(acc1[0], n)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    if args.distributed:
        dist.barrier()
        top1.all_reduce()
        top5.all_reduce()
        for item in acc_percls:
            item.all_reduce()

    progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, args):
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    filename = os.path.join(log_dir, 
                    '{}_{}_checkpoint_{}epochs.pth.tar'.format(args.arch, args.dataset, args.epochs))
    torch.save(state, filename)

    if is_best:
        best_name = os.path.join(log_dir, 
                    '{}_{}_best_{}epochs.pth.tar'.format(args.arch, args.dataset, args.epochs))
        shutil.copyfile(filename, best_name)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / (batch_size)))
        res.append(batch_size)
        return res

def eval_best_model(model, data, criterion, task_content, args):
    resume = os.path.join(args.log_dir, '{}_{}_best_{}epochs.pth.tar'.format(args.arch, args.dataset, args.epochs))
    if args.gpu is None:
            checkpoint = torch.load(resume)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(resume, map_location=loc)

    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {}) for test"
            .format(resume, checkpoint['epoch']))
    del checkpoint
    validate(data["test"] if "test" in data else data["val"], model, criterion, task_content, args)


def build_criterion(args, **kwargs):
    criterion_cfg = args.criterion
    criterion_type = criterion_cfg.pop("type")
    criterion_cls = getattr(loss, criterion_type)
    criterion_cfg.update(kwargs)
    criterion = criterion_cls(**criterion_cfg).to(args.gpu)
    # criterion = ACSL(score_thr=0.7)
    # criterion = nn.BCEWithLogitsLoss(weight=torch.ones(num_classes), reduction='mean').cuda(args.gpu)
    return criterion

if __name__ == '__main__':
    main()
