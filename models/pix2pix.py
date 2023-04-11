import torch
from torch import nn
from models.layers import CategoricalConditionalBatchNorm
import numpy as np
import torch.nn.functional as F


#from pl_bolts.utils.stability import under_review


#@under_review()
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
class Generator_pix2pix_asymmetric(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, in_dim=64, out_dim=64, transform_input = True, num_cond_skip_ch = 32):
        """Paper details:
        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        """
        super().__init__()
        num_downsamples = int(np.log2(in_dim))
        # encoder/donwsample convs
        # self.encoders = [
        #     DownSampleConv(in_channels, 64, batchnorm=False),  # bs x 64 x 128 x 128
        #     DownSampleConv(64, 128),  # bs x 128 x 64 x 64
        #     DownSampleConv(128, 256),  # bs x 256 x 32 x 32
        #     DownSampleConv(256, 512),  # bs x 512 x 16 x 16
        #     DownSampleConv(512, 512),  # bs x 512 x 8 x 8
        #     DownSampleConv(512, 512),  # bs x 512 x 4 x 4
        #     DownSampleConv(512, 512),  # bs x 512 x 2 x 2
        #     DownSampleConv(512, 512, batchnorm=False),  # bs x 512 x 1 x 1
        # ]
        self.transform_input = transform_input
        self.in_dim = in_dim
        self.dim_to_channels = {}
        curr_dim = in_dim
        self.dim_to_channels[curr_dim] = in_channels
        curr_ch = in_channels
        next_ch = 64

        if self.transform_input:
            self.t_inp = nn.Conv2d(in_channels, num_cond_skip_ch, 1, padding=0)
            self.dim_to_channels[curr_dim] = num_cond_skip_ch
            next_ch = 64
        
        
        
        self.encoders = [
            DownSampleConv(curr_ch, next_ch, batchnorm=False)
        ]
        curr_dim = curr_dim//2
        self.dim_to_channels[curr_dim] = next_ch
        curr_ch = next_ch
        
        for i in range(num_downsamples-1):
            if(next_ch <= 256):
                next_ch = curr_ch+32
            if(i < num_downsamples-2):
                self.encoders.append(DownSampleConv(curr_ch, next_ch))
            else:
                self.encoders.append(DownSampleConv(curr_ch, next_ch, batchnorm=False))
            
            curr_dim = curr_dim//2
            self.dim_to_channels[curr_dim] = next_ch
            curr_ch = next_ch
        
        # decoder/upsample convs
        # self.decoders = [
        #     UpSampleConv(512, 512, dropout=True),  # bs x 512 x 2 x 2
        #     UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4
        #     UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
        #     UpSampleConv(1024, 512),  # bs x 512 x 16 x 16
        #     UpSampleConv(1024, 256),  # bs x 256 x 32 x 32
        #     UpSampleConv(512, 128),  # bs x 128 x 64 x 64
        #     UpSampleConv(256, 64),  # bs x 64 x 128 x 128
        # ]

        assert curr_dim==1, "Network Broken"

        num_upsamples = int(np.log2(out_dim))

        self.decoders = [
            UpSampleConv(curr_ch, curr_ch, dropout=True)]  # bs x 512 x 2 x 2
            # UpSampleConv(curr*2, curr, dropout=True),  # bs x 512 x 4 x 4
            # UpSampleConv(curr*2, curr, dropout=True)  # bs x 512 x 8 x 8
            # UpSampleConv(1024, 512),  # bs x 512 x 16 x 16
            # UpSampleConv(1024, 256),  # bs x 256 x 32 x 32
            # UpSampleConv(512, 128),  # bs x 128 x 64 x 64
            # UpSampleConv(256, 64),  # bs x 64 x 128 x 128

        curr_dim*=2
        input_skip = curr_ch+self.dim_to_channels[curr_dim]
        self.decoders.append(UpSampleConv(input_skip, curr_ch, dropout=True))
        
        curr_dim*=2
        input_skip = curr_ch+self.dim_to_channels[curr_dim]

        for i in range(num_upsamples-2):
            if next_ch >= 96:
                next_ch = next_ch-32

            self.decoders.append(UpSampleConv(input_skip, next_ch))
            curr_ch = next_ch
            if curr_dim*2 in self.dim_to_channels:
                curr_dim*=2
            
            if i == num_upsamples-4:
                input_skip = curr_ch
            else:
                input_skip = curr_ch+self.dim_to_channels[curr_dim]
            


        #self.final_conv = nn.ConvTranspose2d(next_ch, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.final_conv = nn.Conv2d(next_ch, out_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x, noise=None):
        
        h = x
        if noise != None:
            h = x = torch.cat((x, noise), axis=1)

        if self.transform_input:
            skip = self.t_inp(h)
        skips_cons = [skip]

        for encoder in self.encoders:
            x = encoder(x)

            skips_cons.append(x)

        skips_cons = skips_cons[:-1]
        count_skips = len(skips_cons)
        curr_dim = 1

        for i, decoder in enumerate(self.decoders[:-1]):
            #print("x_in", x.size())
            x = decoder(x)
            #print("x_out", x.size())
            curr_dim*=2
            if count_skips>0:
                count_skips += (-1)
                skip = skips_cons[count_skips]
            
            else:
                skip = skips_cons[0]
                scale_factor = curr_dim//self.in_dim
                skip = F.upsample(skip, scale_factor)
            
            #print("skip", skip.size())
            if(i < len(self.decoders)-2):
                x = torch.cat((x, skip), axis=1)
                #print("concat", x.size())

        x = self.decoders[-1](x)
        # x = self.decoders[-1](x)
        x = self.final_conv(x)
        return self.tanh(x)


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