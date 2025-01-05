import logging
import os
import datetime
import yaml

import torch

def create_log_dir(args):
    print("=> creating log dir: ./log/{}/{}/{}".format(
                            args.dataset, args.arch, args.train_mode["type"]))
    ISOTIMEFORMAT = '%Y.%m.%d-%H.%M.%S'
    thetime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    log_dir = os.path.join("./log", args.dataset, args.arch, 
                                args.train_mode["type"], thetime)    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    args.log_dir = log_dir
    return log_dir

def copy_config(cfg, args):
    for k in cfg.keys():
        cfg[k] = getattr(args, k)
    
    f_name = os.path.join(args.log_dir, "config.yaml")
    with open(f_name, "w") as f:
        yaml.safe_dump(cfg, f)    
    return

def get_root_logger(log_level=logging.INFO, log_dir='./'):
    ISOTIMEFORMAT = '%Y.%m.%d-%H.%M.%S'
    thetime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    logname = os.path.join(log_dir, thetime + '.log')
    logger = logging.getLogger()

    if not logger.hasHandlers():
        fmt ='%(asctime)s - %(levelname)s - %(message)s'
        format_str = logging.Formatter(fmt)
        logging.basicConfig(filename=logname, filemode='a', format=fmt, level=log_level)
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        logger.addHandler(sh)
    
    logger.setLevel(log_level)
    return logger

def logger_info(logger, dist, info):
    # to only write on rank0
    if not dist:
        logger.info(info)
    else:
        local_rank = torch.distributed.get_rank()
        if local_rank == 0:
            logger.info(info)