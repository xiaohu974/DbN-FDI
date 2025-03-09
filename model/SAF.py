import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numbers
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath, to_2tuple, Mlp


def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) + 1j*(fr(input.imag)+fi(input.real)).type(dtype)

class ComplexConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self,input):
        return apply_complex(self.conv_r, self.conv_i, input)
    
class ComplexELU(nn.Module):
    def __init__(self,):
        super(ComplexELU, self).__init__()
        self.relu_R = nn.ELU()
        self.relu_I = nn.ELU()

    def forward(self, input):
        input_R = input.real
        input_I = input.imag
        output_R = self.relu_R(input_R)
        output_I = self.relu_R(input_I)
        output = output_R.type(torch.complex64) + 1j * output_I.type(torch.complex64)
        return output  
        
    
class FrequencyBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, kernel_size=3,
                 drop=0., act_layer=ComplexELU, init_values=1e-4):
        super().__init__()
        hidden_dim = in_dim 
        out_dim = in_dim
        self.conv1 = nn.Sequential(
            ComplexConv2d(in_dim, hidden_dim, 1),
            act_layer()
        )
        self.conv2 = nn.Sequential(
            ComplexConv2d(hidden_dim, hidden_dim, kernel_size, padding=1, groups=hidden_dim, bias=False),
            act_layer()
        )
        self.conv3 = nn.Sequential(
            ComplexConv2d(hidden_dim, out_dim, 1),
            act_layer()
        )
        self.weight = nn.Parameter(init_values * torch.ones((1, in_dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        shorcut = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.weight*x
        x = x + shorcut
        return x        

## ELA
class ELA(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(12, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.conv1x1(self.pool_h(x).reshape((b, c, h))).reshape((b, c, h, 1))
        x_w = self.conv1x1(self.pool_w(x).reshape((b, c, w))).reshape((b, c, 1, w))
        return x * x_h * x_w        
    

class SpatialBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = ELA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

 
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)
    

class CrossFusionBlock0(nn.Module):
    def __init__(self, dim, num_heads, bias, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, init_values=1e-4):
        super(CrossFusionBlock0, self).__init__()  
 
        self.channel_attention = ChannelAttention(in_channels = dim, reduction = num_heads)

        self.fusion_conv = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(dim // 2, dim)
        self.act = act_layer()
        
        self.out_cov = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
    
    def forward(self, x_s, x_f):
        fusion_out = torch.cat([x_s, x_f], dim=1)
        fusion_out = self.fusion_conv(fusion_out)
        fusion_out = self.norm(fusion_out)
        fusion_out = self.act(fusion_out)

        att_out = self.channel_attention(fusion_out)
        
        out = self.out_cov(att_out)
        out = self.norm(fusion_out)
        return out
    
    
class CrossFusionBlock1(nn.Module):
    def __init__(self, dim, num_heads, bias, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, init_values=1e-4):
        super(CrossFusionBlock1, self).__init__()
        self.channel_attention = ChannelAttention(in_channels = dim, reduction = num_heads)

        self.fusion_conv = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(dim // 3, dim)
        self.act = act_layer()
        
        self.out_cov = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
    
    def forward(self, x_fu, x_s, x_f):
        fusion_out = torch.cat([x_fu, x_s, x_f], dim=1)
        fusion_out = self.fusion_conv(fusion_out)
        fusion_out = self.norm(fusion_out)
        fusion_out = self.act(fusion_out)

        att_out = self.channel_attention(fusion_out)
        
        out = self.out_cov(att_out)
        out = self.norm(fusion_out)
        return out


class DeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups, kernel_size=(3,3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()
        
        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)
        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=True)
    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out       

    

class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=3, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=1, padding=0, bias=bias),
            nn.GELU()
        )
        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))
           
        self.softmax = nn.Softmax(dim=1)

        self.res_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=bias)

    def forward(self, inp_feats):
        batch_size, n_feats, H, W = inp_feats[1].shape

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)

        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V

class ChannelFusionBlock(nn.Module):
    def __init__(self, dim):
        super(ChannelFusionBlock, self).__init__()
        self.skff = SKFF(dim)
        self.Dconv = DeformConv(dim, dim, groups=3, kernel_size=(3,3), stride=1, padding=1, bias=True)
    
    def forward(self, x_cf, x_s, x_f):        
        fusion_out = self.skff([x_cf, x_s, x_f])
        fusion_out = self.Dconv(fusion_out)
        return fusion_out



class down_conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels//2, kernel_size=1)
        )
 
    def forward(self, x):
        return self.down(x)  
    
class down_fconv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = nn.Sequential(
            ComplexConv2d(channels, channels//2, kernel_size=1)
        )
 
    def forward(self, x):
        return self.down(x)  

        
class SAFNet(nn.Module):
    def __init__(self):
        super(SAFNet, self).__init__()

        self.spatial_1 = SpatialBlock(d_model = 96)
        self.frequency_1 = FrequencyBlock(in_dim = 96)

        self.spatial_2 = SpatialBlock(d_model = 48)
        self.frequency_2 = FrequencyBlock(in_dim = 48)

        self.spatial_3 = SpatialBlock(d_model = 24)
        self.frequency_3 = FrequencyBlock(in_dim = 24)
        self.spatial_4 = SpatialBlock(d_model = 12)
        self.frequency_4 = FrequencyBlock(in_dim = 12)
        self.s_down_1 = down_conv(channels = 96)
        self.f_down_1 = down_fconv(channels = 96)
        self.s_down_2 = down_conv(channels = 48)
        self.f_down_2 = down_fconv(channels = 48)
        self.s_down_3 = down_conv(channels = 24)
        self.f_down_3 = down_fconv(channels = 24)
        self.fusion_1 = CrossFusionBlock0(dim = 96, num_heads = 8, bias=False)
        self.fusion_2 = CrossFusionBlock1(dim = 48, num_heads = 4, bias=False)
        self.fusion_3 = CrossFusionBlock1(dim = 24, num_heads = 2, bias=False)
        self.fusion_4 = CrossFusionBlock1(dim = 12, num_heads = 1, bias=False)
        self.S_conv1 = nn.Conv2d(3, 96, 3, 1, 1)
        self.F_conv1 = ComplexConv2d(3, 96, 3, 1, 1)
        self.conv2 = nn.Conv2d(12, 3, 3, 1, 1)
        self.D_conv = DeformConv(in_channels =12, out_channels=3, groups=3, kernel_size=(3,3))
        self.ChannelFusion = ChannelFusionBlock(dim = 3)

    def forward(self, img):
        print("img", img)
        x_fft = torch.fft.rfftn(img, dim=[-2, -1])

        S_1 = self.spatial_1(self.S_conv1(img))
        F_1 = self.frequency_1(self.F_conv1(x_fft))
        FU_1 = self.fusion_1(S_1, torch.fft.irfftn(F_1, dim=[-2, -1]))

        S_1_down = self.s_down_1(S_1)
        F_1_down = self.f_down_1(F_1)
        FU_1_down = self.s_down_1(FU_1)

        S_2 = self.spatial_2(S_1_down)
        F_2 = self.frequency_2(F_1_down)
        FU_2 = self.fusion_2(FU_1_down, S_2, torch.fft.irfftn(F_2, dim=[-2, -1]))

        S_2_down = self.s_down_2(S_2)
        F_2_down = self.f_down_2(F_2)
        FU_2_down = self.s_down_2(FU_2)

        S_3 = self.spatial_3(S_2_down)
        F_3 = self.frequency_3(F_2_down)
        FU_3 = self.fusion_3(FU_2_down, S_3, torch.fft.irfftn(F_3, dim=[-2, -1]))

        S_3_down = self.s_down_3(S_3)
        F_3_down = self.f_down_3(F_3)
        FU_3_down = self.s_down_3(FU_3)

        S_4 = self.spatial_4(S_3_down)
        F_4 = self.frequency_4(F_3_down)
        FU_4 = self.fusion_4(FU_3_down, S_4, torch.fft.irfftn(F_4, dim=[-2, -1]))

        S_out = self.D_conv(S_4)
        F_out = self.D_conv(torch.fft.irfftn(F_4, dim=[-2, -1]))
        FU_out = self.D_conv(FU_4)
        
        out = self.ChannelFusion(FU_out, S_out, F_out)


        return out, S_out, F_out
       

if __name__ == '__main__':
    model = SAFNet()
    pre = model(input)

  