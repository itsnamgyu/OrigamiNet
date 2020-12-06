import os
import sys
import time
import random
import string
import argparse
from collections import namedtuple
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch import autograd
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel as pDDP
import torch
from torchvision import datasets, transforms
import helper
from torchsummary import summary
from torchvision.utils import save_image
import horovod.torch as hvd
import gin

import numpy as np
from tqdm import tqdm, trange
from PIL import Image


import wandb
import ds_load

from utils import CTCLabelConverter, Averager, ModelEma, Metric
from cnv_model import OrigamiNet, ginM
from test import validation

parOptions = namedtuple('parOptions', ['DP', 'DDP', 'HVD'])
parOptions.__new__.__defaults__ = (False,) * len(parOptions._fields)

pO = None
OnceExecWorker = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_bn(model):
    if type(model) in [torch.nn.InstanceNorm2d, torch.nn.BatchNorm2d]:
        init.ones_(model.weight)
        init.zeros_(model.bias)

    elif type(model) in [torch.nn.Conv2d]:
        init.kaiming_uniform_(model.weight)
    
def WrkSeeder(_):
    return np.random.seed((torch.initial_seed()) % (2 ** 32))

@gin.configurable
def train(opt, AMP, WdB, train_data_path, train_data_list, test_data_path, test_data_list, experiment_name, 
            train_batch_size, val_batch_size, workers, lr, valInterval, num_iter, wdbprj, continue_model=''):

    HVD3P = pO.HVD or pO.DDP

    val_batch_size = 1

    if OnceExecWorker and WdB:
        wandb.init(project=wdbprj, name=experiment_name)
        wandb.config.update(opt)
    
    train_dataset = ds_load.myLoadDS(train_data_list, train_data_path)
    valid_dataset = ds_load.myLoadDS(test_data_list, test_data_path , ralph=train_dataset.ralph)

    if opt.num_gpu > 1:
        workers = workers * ( 1 if HVD3P else opt.num_gpu )

    if HVD3P:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=opt.world_size, rank=opt.rank)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=opt.world_size, rank=opt.rank)

    train_loader  = torch.utils.data.DataLoader( train_dataset, batch_size=train_batch_size, shuffle=False, 
                    pin_memory = True, num_workers = int(workers),
                    sampler = train_sampler if HVD3P else None,
                    worker_init_fn = WrkSeeder,
                    collate_fn = ds_load.SameTrCollate
                )
    valid_loader  = torch.utils.data.DataLoader( valid_dataset, batch_size=val_batch_size , pin_memory=True, 
                    num_workers = int(workers), sampler=valid_sampler if HVD3P else None, shuffle=False)
    
    model = OrigamiNet()
    model.apply(init_bn)
    model.train()

    biparams    = list(dict(filter(lambda kv: 'bias'     in kv[0], model.named_parameters())).values())
    nonbiparams = list(dict(filter(lambda kv: 'bias' not in kv[0], model.named_parameters())).values())

    if not pO.DDP:
        model = model.to(device)
    else:
        model.cuda(opt.rank)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=10**(-1/90000))

    if OnceExecWorker and WdB:
        wandb.watch(model, log="all")

    if pO.HVD:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    
    if pO.DDP and opt.rank!=0:
        random.seed()
        np.random.seed()

    if AMP:
        model, optimizer = amp.initialize(model, optimizer, opt_level = "O1")
    if pO.DP:
        model = torch.nn.DataParallel(model)
    elif pO.DDP:
        model = pDDP(model, device_ids=[opt.rank], output_device=opt.rank,find_unused_parameters=False)

    
    
    model_ema = ModelEma(model)

    if continue_model != '':
        checkpoint = torch.load(continue_model, map_location=f'cuda:{opt.rank}' if HVD3P else None)
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        model_ema._load_checkpoint(continue_model, f'cuda:{opt.rank}' if HVD3P else None)

    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)
    converter = CTCLabelConverter(train_dataset.ralph.values())

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = 1e+6
    best_CER = 1e+6
    i = 0
    gAcc = 1
    epoch = 1
    btReplay = False and AMP
    max_batch_replays = 1
    data_dir = '/home/itsnamgyu/test'
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ds_load.myLoadDS('/home/itsnamgyu/test/random_list','/home/itsnamgyu/test/random_test/')

    if HVD3P:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=opt.world_size, rank=opt.rank)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=opt.world_size, rank=opt.rank)

    dataloader  = torch.utils.data.DataLoader(dataset, batch_size=1 , pin_memory=True, 
                    num_workers = int(workers), sampler=None)

    d = iter(dataloader)
    for i in range(0,51) : 
        model.zero_grad()
        image_tensors, labels = next(d)
        image = image_tensors.to(device)
        save_image(image, 'img{}.png'.format(i))
        batch_size = 1
        preds = model(image,'')
        preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)
        preds = preds.permute(1, 0, 2).log_softmax(2)
        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        result = converter.decode(preds_index.data, preds_size.data)
        del preds, image, image_tensors, _, preds_index
        print(" {} ".format(i).center(80, "#"))
        print(result)

                    
                    

def gInit(opt):
    global pO, OnceExecWorker
    gin.parse_config_file(opt.gin)
    pO = parOptions(**{ginM('dist'):True})

    if pO.HVD:
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())

    OnceExecWorker = (pO.HVD and hvd.rank() == 0) or (pO.DP)
    cudnn.benchmark = True


def rSeed(sd):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gin', help='Gin config file')

    opt = parser.parse_args()
    gInit(opt)
    opt.manualSeed = ginM('manualSeed')
    opt.port = ginM('port')

    if OnceExecWorker:
        rSeed(opt.manualSeed)

    opt.num_gpu = torch.cuda.device_count()
    

    if pO.HVD:
        opt.world_size = hvd.size()
        opt.rank       = hvd.rank()
    
    train(opt)
    