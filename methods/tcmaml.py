# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from itertools import chain

class TCMAML(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, approx=True):
        super(TCMAML, self).__init__(model_func, n_way, n_support, change_way=False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)
        
        self.n_task = 4
        self.task_update_num = 5
        self.train_lr = 0.01
        self.approx = approx  # first order approx. default: True   
        self.T = 1  # temperature scaling factor

    def forward(self, x):
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def set_forward(self, x):
        x = x.cuda()
        x_var = Variable(x)
        x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:])  # support data 
        x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:])  # query data
        y_a_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_support))).cuda()  # label for support data

        fast_parameters = list(self.parameters())
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        for task_step in range(self.task_update_num):
            out = self.feature.forward(x_a_i)
            scores = self.classifier.forward(out)
            
            set_loss = self.loss_fn(scores, y_a_i)
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)  # build full graph support gradient of gradient
            if self.approx:
                grad = [g.detach()  for g in grad]  # do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py 
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k]  # create weight.fast 
                else:
                    weight.fast = weight.fast - self.train_lr * grad[k]
                fast_parameters.append(weight.fast)

        out = self.feature.forward(x_b_i)  # average features with query examples
        scores = self.classifier.forward(out)
        out = out.contiguous().view(self.n_way, self.n_query, -1)
        mean_features = out.mean(dim=1)
        
        return scores, mean_features
        
    def measure_uncertainty(self, mean_features):
        '''
        Compute task uncertainty with class-wise average features.
        Each element of the similarity matrix is cosine similarity.
        '''
        similarity_matrix = torch.zeros(self.n_way, self.n_way).cuda()
        similarity_matrix.fill_diagonal_(1)
        for row in range(self.n_way):
            for col in range(row+1, self.n_way):
                cosine_similarity = torch.dot(mean_features[row], mean_features[col])/(torch.norm(mean_features[row], p=2)*torch.norm(mean_features[col], p=2))
                similarity_matrix[row,col] = cosine_similarity
                similarity_matrix[col,row] = cosine_similarity
                
        task_uncertainty = torch.norm(similarity_matrix, p='fro')
        
        return task_uncertainty

    def set_forward_loss(self, x):
        scores, mean_features = self.set_forward(x)
        y_b_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query))).cuda()
        loss = self.loss_fn(scores, y_b_i)
        task_uncertainty = self.measure_uncertainty(mean_features)

        return loss, task_uncertainty
        
    def train_loop(self, epoch, train_loader, optimizer, logfile): 
        print_freq = 10
        avg_loss=0
        task_count = 0
        loss_all = []
        task_uncertainty_all = []
        uncertainty = []
        optimizer.zero_grad()

        for i, (x,_) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0)

            loss, task_uncertainty = self.set_forward_loss(x)
            avg_loss = avg_loss + loss.item()
            loss_all.append(loss)
            uncertainty.append(task_uncertainty.item())
            task_uncertainty_all.append(task_uncertainty)

            task_count += 1
            # TCMAML updates several aggregated tasks at one time
            if task_count == self.n_task:  
                #task_uncertainty_all = torch.reciprocal(nn.functional.softmax(torch.stack(task_uncertainty_all)/self.T))
                # linear scaling (LS)
                #task_uncertainty_all = torch.reciprocal(torch.stack(task_uncertainty_all)/torch.sum(torch.stack(task_uncertainty_all)))
                task_uncertainty_all = nn.functional.softmax((-1) * torch.stack(task_uncertainty_all) / self.T)

                loss_q = torch.mul(task_uncertainty_all, torch.stack(loss_all)).sum(0)
                loss_q.backward()
                optimizer.step()
                
                task_count = 0
                loss_all = []
                task_uncertainty_all = []
                
            optimizer.zero_grad()
            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
                
        avg_uncertainty = np.array(uncertainty).mean()
        logfile.write('Epoch {}, Train-Uncertainty {:.4f} \n'.format(epoch+1, avg_uncertainty))
                      
    def test_loop(self, test_loader, return_std=False): 
        acc_all = []
        task_uncertainty_all = []
        
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0)
            correct_this, count_this, task_uncertainty = self.correct(x)
            task_uncertainty_all.append(task_uncertainty.item())
            acc_all.append(correct_this/count_this *100)

        avg_uncertainty = np.array(task_uncertainty_all).mean()
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('{} Test Acc = {:4.2f}% +/- {:4.2f}%, Valid-Uncertainty {:4.4f}' .format(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num), avg_uncertainty))
        
        if return_std:
            # calibration result
            acc_B = np.array([self.acc_count[i] / self.bin_count[i] if self.bin_count[i] > 0 else 0 for i in range(10)])
            conf_B = np.array([self.conf_count[i] / self.bin_count[i] if self.bin_count[i] > 0 else 0 for i in range(10)])
            ECE = (self.bin_count / self.bin_count.sum() *abs(acc_B - conf_B)).sum()
            MCE = abs(acc_B - conf_B).max()
        
            print('Correct/Total: {}/{}'.format(self.acc_count.sum(), self.bin_count.sum()))
            print('Bin Accuracy: ' + str(acc_B))
            print('ECE: {:.5f}, MCE: {:.5f}'.format(ECE, MCE))
            
            return acc_mean, acc_std
        
        else:
            return acc_mean, avg_uncertainty

