import argparse
import torch
import os

parser = argparse.ArgumentParser()

parser.add_argument('--v_hazy_dir', default='/wspace/Video_Dehazing_Dataset/final-data')
parser.add_argument('--u_hazy_dir', default='')
parser.add_argument('--clean_dir', default='')

parser.add_argument('--checkpoints_dir', default='./checkpoints/')

parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_images', type=int, default=7500)
parser.add_argument('--learning_rate_cc', type=float, default=1e-04)
parser.add_argument('--end_epoch', type=int, default=500)
parser.add_argument('--img_extension', default='.png')

parser.add_argument('--beta1', type=float ,default=0.5)
parser.add_argument('--beta2', type=float ,default=0.999)
parser.add_argument('--wd_g', type=float ,default=0.00005)
parser.add_argument('--wd_d', type=float ,default=0.00000)

parser.add_argument('--batch_cc_l1_loss', type=float, default=0.0)
parser.add_argument('--total_cc_l1_loss', type=float, default=0.0)

# parser.add_argument('--lambda_l1', type=float, default=1.0)

# parser.add_argument('--batch_G_loss', type=float, default=0.0)
# parser.add_argument('--total_G_loss', type=float, default=0.0)

parser.add_argument('--testing_epoch', type=int, default=1)
parser.add_argument('--testing_start', type=int, default=1)
parser.add_argument('--testing_end', type=int, default=1)
parser.add_argument('--testing_mode', default="Syn")

opt = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if not os.path.exists(opt.checkpoints_dir):
    os.makedirs(opt.checkpoints_dir)
