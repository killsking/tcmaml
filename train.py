'''
The source codes are heavily relied on "https://github.com/wyharveychen/CloserLookFewShot".

Before running the code, first prepare datasets. 
In the filelists folder, there are 'download_miniImagenet.sh' and 'download_CUB.sh'. Execute both.
To train TCMAML you should run this file, 'train.py'.
The following command lines will reproduce the results:

miniImagenet 5-way 5-shot
---------------------------
python train.py --dataset miniImagenet --temp 0.2

miniImagenet 5-way 1-shot
---------------------------
python train.py --dataset miniImagenet --n_shot 1 --temp 0.5 

CUB 5-way 5-shot 
------------------
python train.py --dataset CUB --temp 1

CUB 5-way 1-shot 
------------------
python train.py --dataset CUB --n_shot 1 --temp 2

miniImagenet -> CUB (domain shift) 5-way 5-shot
-------------------------------------------------
python train.py --dataset cross --model ResNet18 --temp 2

miniImagenet -> CUB (domain shift) 5-way 1-shot
-------------------------------------------------
python train.py --dataset cross --model ResNet18 --n_shot 1 --temp 2 --stop_epoch 400

miniImagenet 10-way 5-shot w/ Conv6 backbone
----------------------------------------------
python train.py --dataset miniImagenet --model Conv6 --train_n_way 10 --test_n_way 10 --temp 2

To evaluate and check the calibration results after training is done, run 'test.py' with same arguments.
For example,

python test.py --dataset miniImagenet --n_shot 1 --temp 0.5
'''


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import time
import os
import glob

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.tcmaml import TCMAML
from io_utils import model_dict, parse_args, get_resume_file  


def train(train_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0       
    log_dir = os.path.join(params.checkpoint_dir, 'log')
    logfile = open(log_dir, 'w')

    for epoch in range(start_epoch, stop_epoch):
        # train
        model.train()
        model.train_loop(epoch, train_loader, optimizer, logfile)
        model.eval()
        
        # validation
        acc, avg_uncertainty = model.test_loop(val_loader, return_std=False)
        logfile.write('Epoch {}, Valid-Acc {:.4f}, Valid-Uncertainty {:.4f} \n'.format(epoch+1, acc/100, avg_uncertainty))
        
        # save the model with best validation acc
        if acc > max_acc : 
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
            
    logfile.close()
    return model


if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    if params.dataset == 'cross':
        train_file = configs.data_dir['miniImagenet'] + 'all.json' 
        val_file   = configs.data_dir['CUB'] + 'val.json' 
    else:
        train_file = configs.data_dir[params.dataset] + 'base.json' 
        val_file   = configs.data_dir[params.dataset] + 'val.json' 
    
    if 'Conv' in params.model:
        image_size = 84
    else: # ResNet backbone for domain shift
        image_size = 224
    
    if params.model == 'Conv6':
        params.lr = 5e-4
        
    optimization = 'Adam'
    
    # total epoch is stop_epoch * n_task
    if params.stop_epoch == -1: 
        if params.n_shot == 1:
            params.stop_epoch = 600
        elif params.n_shot == 5:
            params.stop_epoch = 400
        else:
            params.stop_epoch = 600
     
    if params.method in ['tcmaml', 'tcmaml_approx']:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        train_datamgr            = SetDataManager(image_size, n_query=n_query, **train_few_shot_params)  # default number of episodes (tasks) is 100 per epoch
        train_loader             = train_datamgr.get_data_loader(train_file, aug=params.train_aug)
         
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot)
        val_datamgr             = SetDataManager(image_size, n_query=n_query, **test_few_shot_params)
        val_loader              = val_datamgr.get_data_loader(val_file, aug=False)        

        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.ResNet.maml = True
        
        model = TCMAML(model_dict[params.model], approx=(params.method == 'tcmaml_approx'), **train_few_shot_params)
        model.T = params.temp
        model.n_task = params.n_task
        
    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    params.checkpoint_dir = '%s/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    params.checkpoint_dir += '_%dway_%dshot' %(params.train_n_way, params.n_shot)
    params.checkpoint_dir += '_temp%2.1f' %(params.temp)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch * model.n_task 

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])

    model = train(train_loader, val_loader,  model, optimization, start_epoch, stop_epoch, params)
