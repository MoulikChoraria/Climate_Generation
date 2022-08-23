import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN
from numpy import sqrt
import torch
from torch.nn.functional import linear  


class Bias(nn.Module):
    def __init__(self, dim):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(dim).unsqueeze(0).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        x += self.bias.repeat(x.size(0), x.size(1), x.size(2), 1)
        return x

class Scalar(nn.Module):
    def __init__(self):
        super(Scalar, self).__init__()
        self.scalar = nn.Parameter(torch.ones(1).unsqueeze(0).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        x *= self.scalar.repeat(x.size(0), x.size(1), x.size(2), x.size(3))
        return x


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflect':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)


    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)



class UNet_2Poly(nn.Module):
    def __init__(self, input_dim, g_layers=[], remove_hot = 0, activation_fn=True, inject_z=True, \
        transform_z=False, concat_injection=False, norm='instance', filter_size = 3, bias = False, skip_connection = False,\
            num_skip=4, skip_size=1, residual=False):
        super(UNet_2Poly, self).__init__()
        
        """Assume symmetric upsampling and downsampling, first and last layer retain same input and output last two dimensions (H X W)"""

        self.residual = residual
        self.total_injections = 0
        self.allowed_injections = int((len(g_layers)-3)/2)*2 - remove_hot
        self.downsampling_layers = int((len(g_layers)-3)/2)
        self.activation_fn = activation_fn
        self.filter_size = filter_size

        #self.downsample_filter = 3
        self.g_layers = g_layers
        self.inject_z = inject_z
        self.num_layers = len(self.g_layers) - 1  # minus the input/output sizes 
        self.transform_z = transform_z
        self.concat_injection = concat_injection
        self.norm = norm
        self.bias = bias
        self.skip_connection = skip_connection
        self.num_skip = num_skip
        self.skip_size = skip_size

        if self.transform_z:
            #for i in range(self.transform_rep):
            setattr(self, "global_transform", nn.Sequential(Bias((input_dim))))

        for i in range(self.num_layers):
                if i < (self.downsampling_layers+1) and i > 0:
                    if self.inject_z and self.allowed_injections > 0:
                        
                        self.total_injections+=1
                        self.allowed_injections-=1

                        if not self.activation_fn:
                            setattr(self, "inject_layer{}".format(i), nn.Sequential(
                                                                    conv(self.g_layers[0], self.g_layers[i+1], kernel_size = self.filter_size, stride=1, bias=self.bias, pad='reflect'),
                                                                    nn.BatchNorm2d(self.g_layers[i+1], affine=True),
                                                                    #nn.AvgPool2d(kernel_size = (2**i+1), stride = 2**i, padding = 1)))
                                                                    nn.MaxPool2d(kernel_size = (2**i+1), stride = 2**i, padding = 1)))
                        else:
                            setattr(self, "inject_layer{}".format(i), nn.Sequential(
                                                                    conv(self.g_layers[0], self.g_layers[i+1], kernel_size = self.filter_size, stride=1, bias=self.bias, pad='reflect'), 
                                                                    nn.BatchNorm2d(self.g_layers[i+1], affine=True),
                                                                    #nn.AvgPool2d(kernel_size = (2**i+1), stride = 2**i, padding = 1),
                                                                    nn.MaxPool2d(kernel_size = (2**i+1), stride = 2**i, padding = 1),
                                                                    nn.LeakyReLU(negative_slope = 0.2)))
                    if not self.activation_fn:
                        if self.norm == 'instance':
                            setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                                    conv(self.g_layers[i], self.g_layers[i+1], kernel_size = self.filter_size, stride = 2, bias=self.bias, pad='reflect'),
                                                                    nn.InstanceNorm2d(self.g_layers[i+1], affine=True),
                                                                    conv(self.g_layers[i+1], self.g_layers[i+1], kernel_size = 1, stride = 1, bias=self.bias, pad='reflect'),
                                                                    nn.InstanceNorm2d(self.g_layers[i+1], affine=True)))  
                        else:
                            setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                                    conv(self.g_layers[i], self.g_layers[i+1], kernel_size = self.filter_size, stride = 2, bias=self.bias, pad='reflect'),
                                                                    nn.BatchNorm2d(self.g_layers[i+1], affine=True),
                                                                    conv(self.g_layers[i+1], self.g_layers[i+1], kernel_size = 1, stride = 1, bias=self.bias, pad='reflect'),
                                                                    nn.BatchNorm2d(self.g_layers[i+1], affine=True)))

                        if self.skip_connection:
                            if self.norm == 'instance':
                                setattr(self, "skip_layer{}".format(i), nn.Sequential(
                                                                    #nn.ConvTranspose2d(2*self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias),
                                                                    conv(self.g_layers[i], self.num_skip, kernel_size = self.skip_size, stride = 1, bias=self.bias, pad='reflect'),
                                                                    nn.InstanceNorm2d(self.num_skip)))
                            else:
                                setattr(self, "skip_layer{}".format(i), nn.Sequential(
                                                                    #nn.ConvTranspose2d(2*self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias),
                                                                    conv(self.g_layers[i], self.num_skip, kernel_size = self.skip_size, stride = 1, bias=self.bias, pad='reflect'),
                                                                    nn.BatchNorm2d(self.num_skip)))                              
                    else:
                        if self.norm == 'instance':
                            setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                                    conv(self.g_layers[i], self.g_layers[i+1], kernel_size = self.filter_size, stride = 2, bias=self.bias, pad='reflect'), 
                                                                    nn.InstanceNorm2d(self.g_layers[i+1]),
                                                                    nn.LeakyReLU(negative_slope = 0.2),
                                                                    conv(self.g_layers[i+1], self.g_layers[i+1], kernel_size = 1, stride = 1, bias=self.bias, pad='reflect'), 
                                                                    nn.InstanceNorm2d(self.g_layers[i+1]),
                                                                    nn.LeakyReLU(negative_slope = 0.2)))
                        else:
                            setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                                    conv(self.g_layers[i], self.g_layers[i+1], kernel_size = self.filter_size, stride = 2, bias=self.bias, pad='reflect'), 
                                                                    nn.BatchNorm2d(self.g_layers[i+1]),
                                                                    nn.LeakyReLU(negative_slope = 0.2),
                                                                    conv(self.g_layers[i+1], self.g_layers[i+1], kernel_size = 1, stride = 1, bias=self.bias, pad='reflect'), 
                                                                    nn.BatchNorm2d(self.g_layers[i+1]),
                                                                    nn.LeakyReLU(negative_slope = 0.2)))

                        if self.skip_connection:
                            if self.norm == 'instance':
                                setattr(self, "skip_layer{}".format(i), nn.Sequential(
                                                                    #nn.ConvTranspose2d(2*self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias),
                                                                    conv(self.g_layers[i], self.num_skip, kernel_size = self.skip_size, stride = 1, bias=self.bias, pad='reflect'),
                                                                    nn.InstanceNorm2d(self.num_skip),
                                                                    nn.LeakyReLU(negative_slope = 0.2)))
                            else:
                                setattr(self, "skip_layer{}".format(i), nn.Sequential(
                                                                    #nn.ConvTranspose2d(2*self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias),
                                                                    conv(self.g_layers[i], self.num_skip, kernel_size = self.skip_size, stride = 1, bias=self.bias, pad='reflect'),
                                                                    nn.BatchNorm2d(self.num_skip),
                                                                    nn.LeakyReLU(negative_slope = 0.2)))


                
                elif i == 0:

                    if not self.activation_fn:
                        if self.norm == 'instance':
                            setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                                    conv(self.g_layers[i], self.g_layers[i+1], kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'), 
                                                                    nn.InstanceNorm2d(self.g_layers[i+1])))
                        else:
                            setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                                    conv(self.g_layers[i], self.g_layers[i+1], kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'), 
                                                                    nn.BatchNorm2d(self.g_layers[i+1])))

                        if self.skip_connection:
                            if self.norm == 'instance':
                                setattr(self, "skip_layer{}".format(i), nn.Sequential(
                                                                    #nn.ConvTranspose2d(2*self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias),
                                                                    conv(self.g_layers[i], self.num_skip, kernel_size = self.skip_size, stride = 1, bias=self.bias, pad='reflect'),
                                                                    nn.InstanceNorm2d(self.num_skip)))
                            else:
                                setattr(self, "skip_layer{}".format(i), nn.Sequential(
                                                                    #nn.ConvTranspose2d(2*self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias),
                                                                    conv(self.g_layers[i], self.num_skip, kernel_size = self.skip_size, stride = 1, bias=self.bias, pad='reflect'),
                                                                    nn.BatchNorm2d(self.num_skip)))


                    else:
                        if self.norm == 'instance':
                            setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                                    conv(self.g_layers[i], self.g_layers[i+1], kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'), 
                                                                    nn.InstanceNorm2d(self.g_layers[i+1]),
                                                                    nn.LeakyReLU(negative_slope = 0.2)))
                        else:
                            setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                                    conv(self.g_layers[i], self.g_layers[i+1], kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'), 
                                                                    nn.BatchNorm2d(self.g_layers[i+1]),
                                                                    nn.LeakyReLU(negative_slope = 0.2)))

                        if self.skip_connection:
                            if self.norm == 'instance':
                                setattr(self, "skip_layer{}".format(i), nn.Sequential(
                                                                    #nn.ConvTranspose2d(2*self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias),
                                                                    conv(self.g_layers[i], self.num_skip, kernel_size = self.skip_size, stride = 1, bias=self.bias, pad='reflect'),
                                                                    nn.InstanceNorm2d(self.num_skip),
                                                                    nn.LeakyReLU(negative_slope = 0.2)))
                            else:
                                setattr(self, "skip_layer{}".format(i), nn.Sequential(
                                                                    #nn.ConvTranspose2d(2*self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias),
                                                                    conv(self.g_layers[i], self.num_skip, kernel_size = self.skip_size, stride = 1, bias=self.bias, pad='reflect'),
                                                                    nn.BatchNorm2d(self.num_skip),
                                                                    nn.LeakyReLU(negative_slope = 0.2)))

                elif i == self.num_layers - 1:
                    if self.skip_connection:
                        curr_layer_filters =  self.g_layers[i] + self.num_skip
                        final_layer_filters = 128 + self.num_skip
                    else:
                        curr_layer_filters =  self.g_layers[i]
                        final_layer_filters = 128


                    if self.norm == 'instance':
                        if not self.activation_fn:
                            setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                            conv(curr_layer_filters, 128, kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'),
                                                            nn.InstanceNorm2d(128)))

                            setattr(self, "conv_layer{}".format(i+1), nn.Sequential(
                                                            conv(final_layer_filters, 128, kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'),
                                                            nn.InstanceNorm2d(128),
                                                            conv(128, self.g_layers[i+1], kernel_size = 1, stride = 1, bias=self.bias, pad='reflect'),
                                                            nn.Sigmoid()))

                        else:
                            setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                            conv(curr_layer_filters, 128, kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'),
                                                            nn.InstanceNorm2d(128),
                                                            nn.LeakyReLU(negative_slope = 0.2)))

                            setattr(self, "conv_layer{}".format(i+1), nn.Sequential(
                                                            conv(final_layer_filters, 128, kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'),
                                                            nn.InstanceNorm2d(128),
                                                            nn.LeakyReLU(negative_slope = 0.2),
                                                            conv(128, self.g_layers[i+1], kernel_size = 1, stride = 1, bias=self.bias, pad='reflect'),
                                                            nn.Sigmoid()))


                    else:
                        if not self.activation_fn:
                            setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                            conv(curr_layer_filters, 128, kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'),
                                                            nn.BatchNorm2d(128)))

                            setattr(self, "conv_layer{}".format(i+1), nn.Sequential(
                                                            conv(final_layer_filters, 128, kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'),
                                                            nn.BatchNorm2d(128),
                                                            conv(128, self.g_layers[i+1], kernel_size = 1, stride = 1, bias=self.bias, pad='reflect'),
                                                            nn.Sigmoid()))

                        else:
                            setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                            conv(curr_layer_filters, 128, kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'),
                                                            nn.BatchNorm2d(128),
                                                            nn.LeakyReLU(negative_slope = 0.2)))

                            setattr(self, "conv_layer{}".format(i+1), nn.Sequential(
                                                            conv(final_layer_filters, 128, kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'),
                                                            nn.BatchNorm2d(128),
                                                            nn.LeakyReLU(negative_slope = 0.2),
                                                            conv(128, self.g_layers[i+1], kernel_size = 1, stride = 1, bias=self.bias, pad='reflect'),
                                                            nn.Sigmoid()))

                else:

                    if self.skip_connection:
                        inject_layer_filters = self.g_layers[i+1]
                        if i > (self.downsampling_layers+1):
                            curr_layer_filters =  self.g_layers[i] + self.num_skip
                        else:
                            curr_layer_filters =  self.g_layers[i]

                    else:
                        curr_layer_filters = self.g_layers[i]
                        inject_layer_filters = self.g_layers[i+1]


                    if self.inject_z and self.allowed_injections > 0:
                        
                        self.total_injections+=1
                        self.allowed_injections-=1

                        if not self.activation_fn:
                            setattr(self, "inject_layer{}".format(i), nn.Sequential(
                                                                    #nn.ConvTranspose2d(self.g_layers[self.downsampling_layers+1], self.g_layers[i+1], 4, 2, 1, bias=self.bias),
                                                                    conv(self.g_layers[self.downsampling_layers+1], inject_layer_filters, kernel_size = self.filter_size, stride=1, bias=self.bias, pad='reflect'),
                                                                    nn.BatchNorm2d(inject_layer_filters, affine=True),
                                                                    #nn.Upsample(scale_factor=2**(i - (self.downsampling_layers+1)), mode='bicubic')))
                                                                    nn.Upsample(scale_factor=2**(i - (self.downsampling_layers)), mode='bicubic')))
                                                                        
                        else:
                            setattr(self, "inject_layer{}".format(i), nn.Sequential(
                                                                    #nn.ConvTranspose2d(self.g_layers[self.downsampling_layers+1], self.g_layers[i+1], 4, 2, 1, bias=self.bias),
                                                                    conv(self.g_layers[self.downsampling_layers+1], inject_layer_filters, kernel_size = self.filter_size, stride=1, bias=self.bias, pad='reflect'),
                                                                    nn.BatchNorm2d(inject_layer_filters, affine=True),
                                                                    #nn.Upsample(scale_factor=2**(i - (self.downsampling_layers+1)), mode='bicubic'),
                                                                    nn.LeakyReLU(negative_slope = 0.2),
                                                                    nn.Upsample(scale_factor=2**(i - (self.downsampling_layers)), mode='bicubic')))

                    if not self.activation_fn:
                        if self.norm == 'instance':
                            setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                                    #nn.ConvTranspose2d(self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias),
                                                                    conv(curr_layer_filters,self.g_layers[i+1], kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'),
                                                                    nn.InstanceNorm2d(self.g_layers[i+1]),
                                                                    conv(self.g_layers[i+1], self.g_layers[i+1], kernel_size = 1, stride = 1, bias=self.bias, pad='reflect'),
                                                                    nn.InstanceNorm2d(self.g_layers[i+1]),
                                                                    nn.Upsample(scale_factor=2.0, mode='bicubic', align_corners=False)))
                        else:
                            setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                                    #nn.ConvTranspose2d(self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias),
                                                                    conv(curr_layer_filters,self.g_layers[i+1], kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'),
                                                                    nn.BatchNorm2d(self.g_layers[i+1]),
                                                                    conv(self.g_layers[i+1], self.g_layers[i+1], kernel_size = 1, stride = 1, bias=self.bias, pad='reflect'),
                                                                    nn.BatchNorm2d(self.g_layers[i+1]),
                                                                    nn.Upsample(scale_factor=2.0, mode='bicubic', align_corners=False)))                            
                           
                    else:
                        if self.norm == 'instance':
                            setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                                    #nn.ConvTranspose2d(self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias),
                                                                    conv(curr_layer_filters,self.g_layers[i+1], kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'), 
                                                                    nn.InstanceNorm2d(self.g_layers[i+1]),
                                                                    nn.LeakyReLU(negative_slope = 0.2),
                                                                    conv(self.g_layers[i+1],self.g_layers[i+1], kernel_size = 1, stride = 1, bias=self.bias, pad='reflect'), 
                                                                    nn.InstanceNorm2d(self.g_layers[i+1]),
                                                                    nn.LeakyReLU(negative_slope = 0.2),
                                                                    nn.Upsample(scale_factor=2.0, mode='bicubic', align_corners=False)))
                        else:
                            setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                                    #nn.ConvTranspose2d(self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias), 
                                                                    conv(curr_layer_filters, self.g_layers[i+1], kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'), 
                                                                    nn.BatchNorm2d(self.g_layers[i+1]),
                                                                    nn.LeakyReLU(negative_slope = 0.2),
                                                                    conv(self.g_layers[i+1], self.g_layers[i+1], kernel_size = 1, stride = 1, bias=self.bias, pad='reflect'), 
                                                                    nn.BatchNorm2d(self.g_layers[i+1]),
                                                                    nn.LeakyReLU(negative_slope = 0.2),
                                                                    nn.Upsample(scale_factor=2.0, mode='bicubic', align_corners=False)))                                                           

    def forward(self, x):

        z = x
        allowed_injections = self.total_injections
        if(self.transform_z):
            z = getattr(self, "global_transform")(x)

        if(self.skip_connection):
            list_skip = []
            point_skip = -1

        for i in range(self.num_layers):
            #print("x",x.size())
            
            #reset polynomial input at upsampling polynomial
            if(i == self.downsampling_layers + 1):
                z = x
                #print("z",z.size())
            
            #save skip layers when downsampling    
            if i < self.downsampling_layers + 1 and self.skip_connection:
                skip = getattr(self, "skip_layer{}".format(i))(x)
                list_skip.append(skip)
                point_skip+=1

            x = getattr(self, "conv_layer{}".format(i))(x)

            #apply injection
            if self.inject_z and allowed_injections > 0:
                if i < self.num_layers - 1 and i > 0:

                    allowed_injections -= 1
                    a = getattr(self, "inject_layer{}".format(i))(z)

                    if self.concat_injection:
                        x = torch.cat((x, a), dim=1)
                    else:

                        if not self.residual:
                            x *= a
                        else: 
                        ### residual
                            x = x*a + x

            #apply skip layers when upsampling
            if i >= self.downsampling_layers + 1 and self.skip_connection and point_skip > -1:

                x = torch.cat((x, list_skip[point_skip]), dim=1)
                point_skip -= 1


        x = getattr(self, "conv_layer{}".format(self.num_layers))(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, d_layers=[], activation_fn=True, spectral_norm=False):
        super(Discriminator, self).__init__()
        self.activation_fn = activation_fn
        self.d_layers = d_layers
        self.num_layers = len(self.d_layers) - 1  # minus the input/output sizes 
        for i in range(self.num_layers):
            if i == 0:
                if not spectral_norm:
                    if not self.activation_fn:
                        setattr(self, "layer{}".format(i), nn.Conv2d(self.d_layers[i], self.d_layers[i+1], 4, 2, 1, bias=False))
                    else:
                        setattr(self, "layer{}".format(i), nn.Sequential(
                                                                nn.Conv2d(self.d_layers[i], self.d_layers[i+1], 4, 2, 1, bias=False),
                                                                nn.LeakyReLU(0.2, inplace=True)))
                else:
                    if not self.activation_fn:
                        setattr(self, "layer{}".format(i), spectral_norm(nn.Conv2d(self.d_layers[i], self.d_layers[i+1], 4, 2, 1, bias=False)))
                    else:
                        setattr(self, "layer{}".format(i), nn.Sequential(
                                                                spectral_norm(nn.Conv2d(self.d_layers[i], self.d_layers[i+1], 4, 2, 1, bias=False)),
                                                                nn.LeakyReLU(0.2, inplace=True)))
            elif i == self.num_layers - 1:
                if not spectral_norm:
                    setattr(self, "layer{}".format(i), nn.Conv2d(self.d_layers[i], self.d_layers[i+1], 4, 2, 1, bias=False))           
                else:
                    setattr(self, "layer{}".format(i), spectral_norm(nn.Conv2d(self.d_layers[i], self.d_layers[i+1], 4, 2, 1, bias=False)))                                            
            else:
                if not spectral_norm:
                    if not self.activation_fn:
                        setattr(self, "layer{}".format(i), nn.Sequential(
                                                                nn.Conv2d(self.d_layers[i], self.d_layers[i+1], 4, 2, 1, bias=False),
                                                                nn.BatchNorm2d(self.d_layers[i+1])))     
                    else:
                        setattr(self, "layer{}".format(i), nn.Sequential(
                                                                nn.Conv2d(self.d_layers[i], self.d_layers[i+1], 4, 2, 1, bias=False),
                                                                nn.BatchNorm2d(self.d_layers[i+1]),
                                                                nn.LeakyReLU(0.2, inplace=True)))                             
                else:
                    if not self.activation_fn:
                        setattr(self, "layer{}".format(i), nn.Sequential(
                                                                spectral_norm(nn.Conv2d(self.d_layers[i], self.d_layers[i+1], 4, 2, 1, bias=False)),
                                                                nn.BatchNorm2d(self.d_layers[i+1])))   
                    else:
                        setattr(self, "layer{}".format(i), nn.Sequential(
                                                                nn.Conv2d(self.d_layers[i], self.d_layers[i+1], 4, 2, 1, bias=False),
                                                                nn.BatchNorm2d(self.d_layers[i+1]),
                                                                nn.LeakyReLU(0.2, inplace=True)))         

    def forward(self, x):
        for i in range(self.num_layers):
            x = getattr(self, "layer{}".format(i))(x)
        return x
