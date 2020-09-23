import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from utils.options.options import opt
from PIL import Image

class ToTensor(object):

    def __call__(self, sample):        

        for k in sample.keys():
            sample[k] = torch.from_numpy(np.array(sample[k]).astype(np.float32)) 
            sample[k] = torch.transpose(torch.transpose(sample[k], 2, 0), 1, 2)
            sample[k] = sample[k]/255.0

        return sample

class Dataset_Load(Dataset):

    def __init__(self, v_hazy_path, u_hazy_path, clean_path, transform=None):
        self.v_hazy_dir = v_hazy_path
        self.u_hazy_dir = u_hazy_path
        self.clean_dir = clean_path
        self.transform = transform

    def __len__(self):
        # get number of frames 
        return len(self.v_hazy_dir)

    def __getitem__(self, index):

        v_hazy_curr_image_name, u_hazy_curr_image_name, clean_curr_image_name = [str(index) + opt.img_extension]*3
        v_hazy_prev_image_name, u_hazy_prev_image_name = [str(index-1) + opt.img_extension]*2
        v_hazy_next_image_name, u_hazy_next_image_name = [str(index+1) + opt.img_extension]*2
        
        v_hazy_curr_im = Image.open(os.path.join(self.v_hazy_dir, v_hazy_curr_image_name))        
        u_hazy_curr_im = Image.open(os.path.join(self.u_hazy_dir, u_hazy_curr_image_name))        
        clean_curr_im = Image.open(os.path.join(self.clean_dir, clean_curr_image_name))

        width, height = v_curr_hazy_im.size

        # if first frame of the video
        # then prev frame assuming zeros
        if index==1:
            v_hazy_prev_im = np.zeros((width, height, 3))
            u_hazy_prev_im = np.zeros((width, height, 3))
        else:
            v_hazy_prev_im = Image.open(os.path.join(self.v_hazy_dir, v_hazy_prev_image_name))        
            u_hazy_prev_im = Image.open(os.path.join(self.v_hazy_dir, u_hazy_prev_image_name))        

        # if last frame then also zeros
        if index == len(self.v_hazy_dir)-1:
            v_hazy_next_im = np.zeros((width, height, 3))
            u_hazy_next_im = np.zeros((width, height, 3))
        else:
            v_hazy_next_im = Image.open(os.path.join(self.v_hazy_dir, v_hazy_next_image_name))        
            u_hazy_next_im = Image.open(os.path.join(self.v_hazy_dir, u_hazy_next_image_name))        
        
        sample = {  
                    'v_curr_hazy': v_hazy_curr_im, 
                    'u_curr_hazy': u_hazy_curr_im,
                    'v_prev_hazy': v_hazy_prev_im, 
                    'u_prev_hazy': u_hazy_prev_im,
                    'v_next_hazy': v_hazy_next_im, 
                    'u_next_hazy': u_hazy_next_im,
                    'clean_curr': clean_curr_im
                 }

        if self.transform != None:
            sample = self.transform(sample)
        return sample