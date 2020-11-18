# import inspect
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
# import utils.data_utils.dataset as dataset

from progress.bar import Bar

from torch.autograd import Function, Variable
from torch.utils.data import DataLoader

from utils.misc.misc import *
from utils.models.models import *
from utils.models.vgg16 import *
from utils.options.options import device, opt
import time
import cv2

testing_epoch = opt.testing_epoch
testing_mode = opt.testing_mode

print("Checking for epoch:", testing_epoch)
print("Checking for mode : ", testing_mode)

CHECKPOINTS_DIR = opt.checkpoints_dir

if testing_mode == "Nat":
    print("Checking for Natural Images")
    HAZY_DIR = "/root/Codes/Dehaze/Test_Nat/hazy/"
else:
    print("Checking for Synthetic Images")
    HAZY_DIR = "/root/Percep_Aug/SOTS/outdoor/hazy/"

result_dir = './EP'+str(testing_epoch)+'_'+testing_mode+'/'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'        

ch = 3
        
netcc = CC_Module().cuda() 
checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR,'netcc_'+ str(testing_epoch)+".pth"))
netcc.load_state_dict(checkpoint['model_state_dict'])
netcc.eval()
netcc.to(device)

if __name__ =='__main__':

    total_files = os.listdir(HAZY_DIR)

    st = time.time()

    for m in total_files:

        print("Testing image ", str(m))

        img=cv2.resize(cv2.imread(HAZY_DIR + str(m)), (512,512))
        img = img.astype(np.float32)
        h,w,c=img.shape

        img=img/255.0

        train_x = np.zeros((1, ch, h, w)).astype(np.float32)

        train_x[0,0,:,:] = img[:,:,0]
        train_x[0,1,:,:] = img[:,:,1]
        train_x[0,2,:,:] = img[:,:,2]

        dataset_torchx = torch.from_numpy(train_x)

        dataset_torchx=dataset_torchx.to(device)

        output=netcc(dataset_torchx)

        # output=output*255.0
        # output = output.cpu()
        # a=output.detach().numpy()

        # res = a[0,:,:,:].transpose((1, 2, 0))

        # cv2.imwrite(result_dir + str(m), np.uint8(res))
        torchvision.utils.save_image(output[0,:,:,:], os.path.join(result_dir + str(m)), normalize=True, scale_each=False)

        print('{')
        print('saved image ', str(m), ' at ', str(result_dir))
        # print('image height ', str(res.shape[1]))
        # print('image width ', str(res.shape[0]))
        print('}\n')
    
    end = time.time()
    print('Total time taken in secs : '+str(end-st))
    print('Per image (avg): '+ str(float((end-st)/len(total_files))))