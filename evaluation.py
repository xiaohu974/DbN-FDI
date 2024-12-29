import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
from utils import PSNR, validation
from model.SAF import SAFNet
from IQA_pytorch import SSIM, MS_SSIM
from data_loaders.lol_v1_new import lowlight_loader_new
from tqdm import tqdm
import cv2
import lpips

import time
from thop import profile

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--img_val_path', type=str, default='./dataset/Endo4IE/Real-Underexp/validation/Underexposed/')
config = parser.parse_args()

print(config)
val_dataset = lowlight_loader_new(images_path=config.img_val_path, mode='test')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

model = SAFNet().cuda()
model.load_state_dict(torch.load("./workdirs/snapshots_folder/best_Epoch_ssim.pth"))
model.eval()


ssim = SSIM()
psnr = PSNR()
lpips = lpips.LPIPS(net='alex').to('cuda')

ssim_list = []
psnr_list = []
lpips_list = []
# mse_list = []
# niqe_list = []

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if config.save:
    result_path = config.img_val_path.replace('Underexposed', 'Result')
    mkdir(result_path)

with torch.no_grad():
    for i, imgs in tqdm(enumerate(val_loader)):
        #print(i)
        low_img, high_img, name = imgs[0].cuda(), imgs[1].cuda(), str(imgs[2][0])
        print(name)
        start = time.time()                
        #print(low_img.shape)
        enhanced_img, S_out, F_out, FU_out = model(low_img)
        
        end_time = (time.time() - start)
        print('infer_time:', end_time)
        

        
        # enhanced_img, _, _ = model(low_img)
        if config.save:
            torchvision.utils.save_image(enhanced_img, result_path + str(name) + '.jpg')

        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_img, high_img).item()
        # mse_value = F.mse_loss(enhanced_img, high_img).item()
        lpips_value = lpips(enhanced_img, high_img).item()
        # image = enhanced_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # niqe_value = niqe(enhanced_img)
     

        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)
        lpips_list.append(lpips_value)
        # mse_list.append(mse_value)
        # niqe_list.append(niqe_value)


SSIM_mean = np.mean(ssim_list)
PSNR_mean = np.mean(psnr_list)
LPIPS_mean = np.mean(lpips_list)
# MSE_mean = np.mean(mse_list)
# NIQE_mean = np.mean(niqe_list)
print('The SSIM Value is:', SSIM_mean)
print('The PSNR Value is:', PSNR_mean)
print('The LPIPS Value is:', LPIPS_mean)
# print('The MSE Value is:', MSE_mean)
# print('The NIQE Value is:', NIQE_mean)
