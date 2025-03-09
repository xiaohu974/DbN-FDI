import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import cv2
import os
import math
from IQA_pytorch import SSIM, MS_SSIM
import torch.distributed as dist
import torch.fft


EPS = 1e-3
PI = 22.0 / 7.0
# calculate PSNR
class PSNR(nn.Module):
    def __init__(self, max_val=0):
        super().__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return 0

        return 10 * torch.log10((1.0 / mse))


def adjust_learning_rate(optimizer, epoch,  lr_decay=0.5):

    # --- Decay learning rate --- #
    step = 20

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))

def get_dist_info():

    # --- Get dist info --- #

    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def visualization(img, img_path, iteration):

    # --- Visualization for Checking --- #
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    img = img.cpu().numpy()

    for i in range(img.shape[0]):
        # save name
        name = str(iteration) + '_' + str(i) + '.jpg'
        print(name)

        img_single = np.transpose(img[i, :, :, :], (1, 2, 0))
        # print(img_single)
        img_single = np.clip(img_single, 0, 1) * 255.0
        img_single = cv2.UMat(img_single).get()
        img_single = img_single / 255.0

        plt.imsave(os.path.join(img_path, name), img_single)

ssim = SSIM()
psnr = PSNR()

def validation(model, val_loader):
    # lpips = lpips.LPIPS(net='alex')

    ssim = SSIM()
    psnr = PSNR()
    ssim_list = []
    psnr_list = []
    lpips_list = []
    for i, imgs in enumerate(val_loader):
        with torch.no_grad():
            low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
            out, S_out, F_out = model(low_img)
            # print(enhanced_img.shape)
        ssim_value = ssim(out, high_img, as_loss=False).item()
        #ssim_value = ssim(enhanced_img, high_img).item()
        psnr_value = psnr(out, high_img).item()
        # print('The %d image SSIM value is %d:' %(i, ssim_value))
        # lpipa_value = lpips(enhanced_img, high_img).item()
        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)
        # lpips_list.append(lpipa_value)

    SSIM_mean = np.mean(ssim_list)
    PSNR_mean = np.mean(psnr_list)
    print('The SSIM Value is:', SSIM_mean)
    print('The PSNR Value is:', PSNR_mean)
    return SSIM_mean, PSNR_mean

def validation_shadow(model, val_loader):

    ssim = SSIM()
    psnr = PSNR()
    ssim_list = []
    psnr_list = []
    for i, imgs in enumerate(val_loader):
        with torch.no_grad():
            low_img, high_img, mask = imgs[0].cuda(), imgs[1].cuda(), imgs[2].cuda()
            _, _, enhanced_img = model(low_img, mask)
            # print(enhanced_img.shape)
        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        #ssim_value = ssim(enhanced_img, high_img).item()
        psnr_value = psnr(enhanced_img, high_img).item()
        # print('The %d image SSIM value is %d:' %(i, ssim_value))
        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)

    SSIM_mean = np.mean(ssim_list)
    PSNR_mean = np.mean(psnr_list)
    print('The SSIM Value is:', SSIM_mean)
    print('The PSNR Value is:', PSNR_mean)
    return SSIM_mean, PSNR_mean


##########Loss Function#########
#  MS_SSIM_L1_LOSS
class MS_SSIM_L1_LOSS(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range = 1.0,
                 K=(0.01, 0.03),
                 alpha=0.025,
                 compensation=200.0,
                 cuda_dev=0,):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation=compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3*len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3*idx+0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM*PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
                               groups=3, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation*loss_mix

        return loss_mix.mean()

#  Charbonnier Loss
class Charbonnier_Loss(nn.Module):
    def __init__(self, eps=1e-3):
        super(Charbonnier_Loss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        diff = input - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss
    
# Spatial Consistency Loss
class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org, enhance):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)
        return torch.mean(E)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, epsilon=1e-8):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, fake, real):
        ## 2D DFT with orthonormalization
        fake_fft = torch.fft.rfftn(fake, norm='ortho')
        real_fft = torch.fft.rfftn(real, norm='ortho')

        x_dist = (real_fft.real - fake_fft.real) ** 2
        y_dist = (real_fft.imag - fake_fft.imag) ** 2

        distance_matrix = torch.sqrt(x_dist + y_dist + self.epsilon) 

        ## squared Euclidean distance
        squared_distance = distance_matrix ** 2

        ## weight for spatial frequency
        weight_matrix = distance_matrix ** self.alpha

        # normalization weight_matrix to [0,1]
        norm_weight_matrix = (weight_matrix - torch.min(weight_matrix)) /  (torch.max(weight_matrix) - torch.min(weight_matrix))

        prod = torch.mul(squared_distance, norm_weight_matrix)

        FFL = torch.sum(prod) / (256 * 256 * 3)

        return FFL 
    
class FFTChiSqLoss(nn.Module):
    def __init__(self):
        super(FFTChiSqLoss, self).__init__()

    def forward(self, low_light, normal_light):
        # 进行傅里叶变换
        low_light_fft = torch.fft.rfftn(low_light, norm='ortho')
        normal_light_fft = torch.fft.rfftn(normal_light, norm='ortho')

        # 取实部和虚部
        input_real = low_light_fft.real
        input_imag = low_light_fft.imag
        target_real = normal_light_fft.real
        target_imag = normal_light_fft.imag

        # 求出差异
        diff_real = torch.abs(input_real - target_real)
        diff_imag = torch.abs(input_imag - target_imag)

        # 取最大值
        max_diff_real, _ = torch.max(diff_real.reshape(diff_real.size(0), -1), dim=1, keepdim=True)
        max_diff_imag, _ = torch.max(diff_imag.reshape(diff_imag.size(0), -1), dim=1, keepdim=True)

        # 计算切比雪夫距离
        chisq_real = F.pairwise_distance(input_real, target_real, p=float('inf')).mean()
        chisq_imag = F.pairwise_distance(input_imag, target_imag, p=float('inf')).mean()

        # 返回差异和切比雪夫距离的平均值
        loss = (chisq_real + chisq_imag) / 2.0
        
        return loss
