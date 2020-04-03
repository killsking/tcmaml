import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time

import configs
import backbone
from data.datamgr import SetDataManager
from methods.tcmaml import TCMAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file

if __name__ == '__main__':
    params = parse_args('test')

    # number of test tasks: 600
    iter_num = 600
    few_shot_params = dict(n_way = params.test_n_way, n_support = params.n_shot) 

    if params.method in ['tcmaml' , 'tcmaml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.ResNet.maml = True
        model = TCMAML(model_dict[params.model], approx=(params.method == 'tcmaml_approx'), **few_shot_params)

    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    checkpoint_dir = '%s/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    checkpoint_dir += '_%dway_%dshot' %(params.train_n_way, params.n_shot)
    checkpoint_dir += '_temp%2.1f' %(params.temp)
    
    if not os.path.isdir(checkpoint_dir):
        raise ValueError('Checkpoint not exists')
    
    if params.save_iter != -1:
        modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
    else:
        modelfile   = get_best_file(checkpoint_dir)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" + str(params.save_iter)
    else:
        split_str = split

    if 'Conv' in params.model:
        image_size = 84
    else:  # ResNet backbone for domain shift
        image_size = 224
    
    # number of query examples per class in testset is 15
    datamgr = SetDataManager(image_size, n_eposide=iter_num, n_query=15, **few_shot_params)
    if params.dataset == 'cross':
        if split == 'base':
            loadfile = configs.data_dir['miniImagenet'] + 'all.json' 
        else:
            loadfile   = configs.data_dir['CUB'] + split + '.json'
    else: 
        loadfile    = configs.data_dir[params.dataset] + split + '.json'
    test_loader     = datamgr.get_data_loader(loadfile, aug=False)
    
    model.eval()
    acc_mean, acc_std = model.test_loop(test_loader, return_std=True)
        
    with open('./record/results.txt', 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime()) 
        aug_str = '-aug' if params.train_aug else ''
        exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.train_n_way, params.test_n_way)
        acc_str = '%d Test Acc = %4.2f%% +/- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
        f.write('Time: %s, Setting: %s, Acc: %s \n' %(timestamp, exp_setting, acc_str))
