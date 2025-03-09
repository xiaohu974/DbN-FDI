import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import os
import argparse
import numpy as np
from data_loaders.lol_v1_new import lowlight_loader_new
from model.SAF import SAFNet
from IQA_pytorch import SSIM
from utils import PSNR, validation, MS_SSIM_L1_LOSS, FFTChiSqLoss, L_spa, FocalLoss, Charbonnier_Loss

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--img_path', type=str, default='./dataset/Endo4IE/Real-Underexp/train/Underexposed/')
parser.add_argument('--img_val_path', type=str, default='./dataset/Endo4IE/Real-Underexp/validation/Underexposed/')

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--pretrain_dir', type=str, default=None)
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--snapshots_folder', type=str, default="workdirs/snapshots_folder/")

config = parser.parse_args()

print(config)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

if not os.path.exists(config.snapshots_folder):
    os.makedirs(config.snapshots_folder)

model = SAFNet().cuda()
if config.pretrain_dir is not None:
    model.load_state_dict(torch.load(config.pretrain_dir))

train_dataset = lowlight_loader_new(images_path=config.img_path)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                           pin_memory=True)
val_dataset = lowlight_loader_new(images_path=config.img_val_path, mode='test')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
  
device = next(model.parameters()).device

criterion = MS_SSIM_L1_LOSS()
Spa = L_spa()

FocalLoss = FocalLoss()
FFTChiSqLoss = FFTChiSqLoss()
Charbonnier_Loss = Charbonnier_Loss()

epoch_losses = []
ssim = SSIM()
psnr = PSNR()
ssim_high = 0
psnr_high = 0

model.train()

for epoch in range(config.num_epochs):
    epoch_loss = 0
    for iteration, imgs in enumerate(train_loader):
        low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
        optimizer.zero_grad()
        model.train()
        out, S_out, F_out = model(low_img)
        loss = Charbonnier_Loss(out, high_img) + FFTChiSqLoss(F_out, high_img) + Spa(S_out, high_img)

        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_losses.append(loss.item())
        if ((iteration + 1) % config.display_iter) == 0:
            print("Loss at iteration", iteration + 1, ":", loss.item())
            with open(config.snapshots_folder + '/loss.txt', 'a+') as f:
                f.write('epoch' + str(epoch) + 'iteration' + str(iteration + 1) + ':' + 'the Loss is  ' + str(loss.item()) + '\n')
            f.close()

    model.eval()
    SSIM_mean, PSNR_mean = validation(model, val_loader)

    with open(config.snapshots_folder + '/log.txt', 'a+') as f:
        f.write('epoch' + str(epoch) + ':' + 'the SSIM is  ' + str(SSIM_mean) + '  the PSNR is  ' + str(PSNR_mean) + '\n')
    
    if SSIM_mean > ssim_high:
        ssim_high = SSIM_mean
        print('the highest SSIM value is:', str(ssim_high))
        torch.save(model.state_dict(), os.path.join(config.snapshots_folder, "best_Epoch_ssim" + '.pth'))
    if PSNR_mean > psnr_high:
        psnr_high = PSNR_mean
        print('the highest PSNR value is:', str(psnr_high))
        torch.save(model.state_dict(), os.path.join(config.snapshots_folder, "best_Epoch_psnr" + '.pth'))
    f.close()
print('the highest SSIM value is:', str(ssim_high))
print('the highest PSNR value is:', str(psnr_high))


with open(config.snapshots_folder + '/log.txt', 'a+') as f:
    f.write('the highest SSIM value is:  ' + str(ssim_high) + '  the highest PSNR value is:  ' + str(psnr_high) + '\n')
f.close()