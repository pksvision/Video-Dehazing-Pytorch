import os

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.options.options import opt


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')

# def getLatestCheckpointName():
#     if os.path.exists(opt.checkpoints_dir):
#         file_names = os.listdir(opt.checkpoints_dir)
#         names_ext = [os.path.splitext(x) for x in file_names]
#         checkpoint_names_G = []    
#         l = []
#         for i in range(len(names_ext)):
#             module = names_ext[i][1] == '.pth' and str(names_ext[i][0]).split('_')
#             if module[0] == 'netcc':
#                 checkpoint_names_G.append(int(module[1]))
#         if len(checkpoint_names_G) == 0 :
#             return None    
#         g_index = max(checkpoint_names_G)    
#         ckp_g = None
#         for i in file_names:    
#             if int(str(i).split('_')[1].split('.')[0]) == g_index and str(i).split('_')[0] == 'netcc':
#                 ckp_g = i
#                 break
#         return ckp_g
