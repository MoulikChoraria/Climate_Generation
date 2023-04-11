import torch
from torch import nn
from models.layers import CategoricalConditionalBatchNorm
import numpy as np
import torch.nn.functional as F


class UpSampleConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel=3, strides=1, padding=1, activation=True, batchnorm=True, n_classes = 0, dropout=False#, use_upsample = True
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.n_clases = n_classes

        # if use_upsample:
        #     self.deconv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, strides, pa), 
        #                                                         #norm_func(input_norm),
        #                                                         nn.AdaptiveAvgPool2d(track_dim),
        #                                                         nn.LeakyReLU(negative_slope = 0.2)) 
        # else:
        #self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding)
        self.deconv = nn.Conv2d(in_channels, out_channels, kernel, stride = strides, padding=padding)

        if batchnorm and n_classes>1:
            self.bn = CategoricalConditionalBatchNorm(out_channels, n_classes)
        else:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, h, y=None):
        #print("in_conv", x.size())
        x = self.deconv(h)
        #print("out_conv", x.size())
        x = F.upsample(x, scale_factor=2)
        #print("up_layer", x.size())

        if self.batchnorm: 
            if self.n_clases > 1:
                x = self.bn(x, y)
            else:
                x = self.bn(x)

        if self.dropout:
            x = self.drop(x)
        return x


#@under_review()
class DownSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True, n_classes=0):
        """Paper details:
        - C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        """
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.n_clases = n_classes

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm and n_classes>1:
            self.bn = CategoricalConditionalBatchNorm(out_channels, n_classes)
        else:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x, y = None):
        x = self.conv(x)
        if self.batchnorm: 
            if self.n_clases > 1:
                x = self.bn(x, y)
            else:
                x = self.bn(x)

        if self.activation:
            x = self.act(x)
        return x


#@under_review()
class PatchGAN_asymmetric(nn.Module):
    def __init__(self, input_dim = 64, input_channels=1, conditional_channels= 1, conditional_dim=64, ch=64, mult_factor=2, add_skip=False):
        super().__init__()
        self.counter=0
        self.add_skip = add_skip

        mf = lambda power: mult_factor ** power
        mf_term = 0

        curr_dim = input_dim
        in_ch = input_channels
        if(curr_dim == conditional_dim):
            in_ch = input_channels + conditional_channels
            counter = 0

        self.d1 = DownSampleConv(in_ch, ch*mf(mf_term), batchnorm=False)
        

        curr_dim = input_dim//2
        in_ch = ch*mf(mf_term)
        if(curr_dim == conditional_dim):
            in_ch = ch*mf(mf_term) + conditional_channels
            counter=1
        mf_term +=1

        self.d2 = DownSampleConv(in_ch, ch*mf(mf_term))
        
        curr_dim = input_dim//2
        in_ch = ch*mf(mf_term)
        if(curr_dim == conditional_dim):
            in_ch = ch*mf(mf_term) + conditional_channels
            counter=2
        mf_term +=1
        self.d3 = DownSampleConv(in_ch, ch*mf(mf_term))
        
        curr_dim = input_dim//2
        in_ch = ch*mf(mf_term)
        if(curr_dim == conditional_dim):
            in_ch = ch*mf(mf_term) + conditional_channels
            counter=3
        
        mf_term +=1
        if self.add_skip:
            in_ch+=1
        self.d4 = DownSampleConv(in_ch, ch*mf(mf_term))


        self.final = nn.Conv2d(ch*mf(mf_term), 1, kernel_size=1)
        self.counter = counter

    def forward(self, x, y):
        #x = torch.cat([x, y], axis=1)
        if self.counter == 0:
            x = torch.cat([x, y], axis=1)
        
        x = self.d1(x)
        if self.counter == 1:
            x = torch.cat([x, y], axis=1)
        x = self.d2(x)
        if self.counter == 2:
            x = torch.cat([x, y], axis=1)
        x = self.d3(x)
        if self.counter == 3:
            x = torch.cat([x, y], axis=1)
        
        if self.add_skip:
            y_skip = F.interpolate(y, size=(x.size(2), x.size(3)), mode='nearest-exact')
            x = torch.cat([x, y_skip], axis=1)

        x = self.d4(x)
        x = self.final(x)
        return x