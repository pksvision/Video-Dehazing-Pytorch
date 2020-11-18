# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr
# import os
# import sys
# import numpy as np
# from PIL import Image
# from statistics import mean 

# # true_path = sys.argv[1]
# # pred_path = sys.argv[2]

# def get_ssim_psnr(true_path, pred_path):

# 	gt_images = os.listdir(str(true_path))
# 	deblured_images = os.listdir(str(pred_path))

# 	p_val =[]
# 	s_val =[]

# 	for img in gt_images:
# 		gt_image = Image.open(os.path.join(true_path, img))
# 		gt_image = np.asarray(gt_image)

# 		pred_image_name = str(img).split("-")[0]+".png"
# 		pred_image = Image.open(os.path.join(pred_path, pred_image_name))
# 		pred_image = np.asarray(pred_image)

# 		p_val.append(psnr(gt_image, pred_image))
# 		s_val.append(ssim( gt_image, pred_image, data_range=255, multichannel=True))

# 	# print("Avg: PSNR " , mean(p_val))
# 	# print("Avg: SSIM " , mean(s_val))
# 	return mean(s_val), mean(p_val)