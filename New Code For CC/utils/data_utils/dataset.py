import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from utils.options.options import opt
from PIL import Image
import h5py

class ToTensor(object):
    def __call__(self, sample):        

        v_hazy, clean = sample['v_curr_hazy'], sample['u_curr_hazy']
        v_hazy, clean = torch.from_numpy(np.float32(v_hazy)), torch.from_numpy(np.float32(clean))
        v_hazy, clean = torch.transpose(torch.transpose(v_hazy, 2, 0), 1, 2), torch.transpose(torch.transpose(clean, 2, 0), 1, 2)
        sample['v_curr_hazy'], sample['u_curr_hazy'] = v_hazy, clean

        return sample

class Dataset_Load(Dataset):
    def __init__(self, v_hazy_path, u_hazy_path, clean_path, transform=None):
        self.v_hazy_dir = v_hazy_path
        self.u_hazy_dir = u_hazy_path
        self.clean_dir = clean_path
        self.transform = transform        

    def __len__(self):
        return opt.num_images

    def __getitem__(self, index):        
        
        filename=os.path.join(self.v_hazy_dir, str(index)+".h5")
        # print(filename)
        f=h5py.File(filename,'r')

        v_hazy_curr_im=f['haze'][:]
        clean_curr_im=f['gt'][:]
        u_hazy_curr = f['uni'][:]
        
        sample = {  
                    'v_curr_hazy': v_hazy_curr_im, 
                    'u_curr_hazy':u_hazy_curr
                    # 'clean_curr': clean_curr_im
                 }

        if self.transform != None:
            sample = self.transform(sample)
        return sample
