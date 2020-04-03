import numpy as np
import os
import glob
import argparse
import backbone


model_dict = dict(Conv4 = backbone.Conv4,
                  Conv6 = backbone.Conv6,
                  ResNet10 = backbone.ResNet10,
                  ResNet18 = backbone.ResNet18) 

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset'     , default='miniImagenet',        help='miniImagenet/CUB/cross')  # cross: miniImagenet -> CUB
    parser.add_argument('--model'       , default='Conv4',      help='model: Conv{4|6} / ResNet{10|18}')  # backbone CNN model
    parser.add_argument('--method'      , default='tcmaml_approx',   help='tcmaml{_approx}') 
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training') 
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation)') 
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of shots (support examples) in each class') 
    parser.add_argument('--train_aug'   , default=True,  help='perform data augmentation or not during training')  # We used data augmentation in the paper.
    parser.add_argument('--temp'        , default=1, type=float, help='temperature scaling factor')
    parser.add_argument('--lr'          , default=0.001, type=float, help='learning rate for meta-update')
    parser.add_argument('--n_task'      , default=4, type=int, help='number of tasks per batch when computing total loss')

    if script == 'train':
        parser.add_argument('--save_freq'   , default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=-1, type=int, help ='Stopping epoch')  # Each epoch contains 100 episodes. The default epoch number is dependent on number of shots. See train.py for details.
        parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        
    elif script == 'test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel')  # base: train, val: validation, novel: test
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved model in x epoch, use the best model if x is -1')
        
    else:
        raise ValueError('Unknown script')
        
    return parser.parse_args()


def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
