# Baseline Code 
# Date : 23/09/2020
# Author : Ira Bisht and PKS
# For : Video De-dehazing

import inspect
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import utils.data_utils.dataset as dataset

from progress.bar import Bar

from torch.autograd import Function, Variable
from torch.utils.data import DataLoader

from utils.misc.misc import *
from utils.models.models import *
from utils.models.vgg16 import *
from utils.options.options import device, opt

torch.manual_seed(191009)


# define class for training the video de-dehazing module
# initialize the params
class Train(nn.Module):
    """docstring for Train"""
    def __init__(self):
        super(Train, self).__init__()
        # define : batch size 
        self.batches = int(opt.num_images / opt.batch_size)

        # define cc module
        self.netcc = CC_Module().cuda()
        self.netcc.apply(init_weights)
        print('****CC network loaded****')
        
        # define loss, optimizer, schedular
        # lets start with first spatial pixel based loss 
        # later over this we can define rest of the lossed from fig 4
        self.cc_l1_loss = nn.SmoothL1Loss()
        
        # define cc optimizer
        self.optim_cc = optim.Adam(self.netcc.parameters(), 
                                  lr=opt.learning_rate_cc, 
                                  betas = (opt.beta1, opt.beta2), 
                                  weight_decay=opt.wd_g)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    # @staticmethod
    def start_training(self):
        
        if not os.path.exists(opt.checkpoints_dir):
            os.makedirs(opt.checkpoints_dir)

        if len(os.listdir(opt.checkpoints_dir)) == 0:
            latest_checkpoint_cc = None
        else:
            latest_cc_model = max[int(name.split('.')[0].split('_')[1]) for name in os.listdir(opt.checkpoints_dir)]   
            latest_checkpoint_cc = 'netcc_'+str(latest_cc_model)+'.pth'

        # print('loading model for cc ', latest_checkpoint_cc)
        
        if latest_checkpoint_cc == None :
            start_epoch = 1
            print('***No checkpoints found for netcc ! retraining***')
        else:
            checkpoint_cc = torch.load(os.path.join(opt.checkpoints_dir, latest_checkpoint_cc))    
            start_epoch = checkpoint_cc['epoch'] + 1
            self.netcc.load_state_dict(checkpoint_cc['model_state_dict'])
            self.optim_cc.load_state_dict(checkpoint_cc['optimizer_state_dict'])
            print('***Restoring model from checkpoint*** ' + str(start_epoch))
            
        self.netcc.train()    
        for epoch in range(start_epoch, opt.end_epoch + 1):

            bar = Bar('Training', max=self.batches)
            opt.total_l1_loss = 0.0
            
            cnt_vid = 0
            hazy_dir = os.listdir(opt.v_hazy_dir)
            num_videos = len(hazy_dir)

            random.shuffle(hazy_dir)
            # print(hazy_dir)
            
            for vid in hazy_dir:

                print("Training for the video ", str(vid))
                prev_est_cnt = 0
                dataset_obj = dataset.Dataset_Load(v_hazy_path = os.path.join(opt.v_hazy_dir, vid), 
                                                   u_hazy_path = os.path.join(opt.u_hazy_dir, vid), 
                                                   clean_path = os.path.join(opt.clean_dir, vid),
                                                   transform = dataset.ToTensor())

                dataloader = DataLoader(dataset_obj, batch_size=1, shuffle=False)


                for i_batch, sample_batched in enumerate(dataloader):
                    
                    v_hazy_curr_batch = sample_batched['v_curr_hazy'].cuda()
                    u_hazy_curr_batch = sample_batched['u_curr_hazy'].cuda()
                    clean_curr_batch = sample_batched['curr_clean'].cuda()

                    v_hazy_prev_batch = sample_batched['v_prev_hazy'].cuda()
                    u_hazy_prev_batch = sample_batched['u_prev_hazy'].cuda()

                    v_hazy_next_batch = sample_batched['v_next_hazy'].cuda()
                    u_hazy_next_batch = sample_batched['u_next_hazy'].cuda()

                    if prev_est_cnt == 0:
                        prev_est_dehazed_batch = torch.zeros_like(v_hazy_curr_batch, dtype=torch.float32).cuda()
                        prev_est_cnt +=1 
                    else:
                        prev_est_dehazed_batch = torch.from_numpy(prev_est_dehazed_batch).to(dtype=torch.float32).cuda()
                    
                    self.optim_cc.zero_grad()
                    
                    cc_v_hazy_curr_batch = self.netcc(v_hazy_curr_batch)
                    cc_v_hazy_prev_batch = self.netcc(v_hazy_prev_batch)
                    cc_v_hazy_next_batch = self.netcc(v_hazy_next_batch)

                    batch_cc_l1_loss = self.cc_l1_loss(cc_v_hazy_curr_batch, u_hazy_curr_batch)
                    batch_cc_l1_loss += self.cc_l1_loss(cc_v_hazy_prev_batch, u_hazy_prev_batch)
                    batch_cc_l1_loss += self.cc_l1_loss(cc_v_hazy_next_batch, u_hazy_next_batch)

                    batch_cc_l1_loss.backward()
                    opt.batch_cc_l1_loss = batch_cc_l1_loss.item()
                    opt.total_cc_l1_loss += opt.batch_cc_l1_loss
                    
                    self.optim_cc.step()

                    
                    
                    # schedule it using total loss
                    bar.suffix = f' Epoch : {epoch} | ({i_batch+1}/{self.batches}) | ETA: {bar.eta_td} | g_l1: {opt.batch_cc_l1_loss}'
                    bar.next()
            print('\nFinished ep. %d, lr = %.6f, total_l1 = %.6f' % (epoch, get_lr(self.optim_cc) opt.total_l1_loss))

            bar.finish()
            torch.save({'epoch':epoch, 
                        'model_state_dict':self.netcc.state_dict(), 
                        'optimizer_state_dict':self.optim_cc.state_dict(), 
                        'l1_loss':opt.total_cc_l1_loss, 
                        'opt':opt
                        }, os.path.join(opt.checkpoints_dir, 'netcc_' + str(epoch) + '.pth'))
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError        

if __name__ == '__main__':
    model = Train()
    _ = model.start_training()
