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


class Conditional_Model(nn.Module):
    def __init__(self, input_dim, num_classes=5, g_layers=[], remove_hot = 0, activation_fn=True, inject_z=True, transform_rep=1, \
                    transform_z=False, concat_injection=False, norm='instance', filter_size = 3, bias = False,\
                            skip_connection = False, num_skip=4, skip_size=1, residual=False, up_mode = 'upsample'):
        super(Conditional_Model, self).__init__()

        self.residual = residual
        self.num_layers = len(g_layers)-1
        self.allowed_injections = (self.num_layers - 1) - remove_hot
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
        self.g_layers = g_layers
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.transform_rep = transform_rep
        self.upsample_mode = up_mode
        self.embedding = nn.Embedding(self.num_classes, self.input_dim)
        
        if self.transform_z:
            for i in range(self.transform_rep):
                setattr(self, "global{}".format(i), nn.Sequential(
                                                        nn.Linear(self.g_layers[0], self.g_layers[0]),
                                                        nn.ReLU()))
        
        total_injections = 0
        
        if(self.norm == 'instance'):
            norm_func = nn.InstanceNorm2d()
        else:
            norm_func = nn.BatchNorm2d()


        for i in range(self.num_layers):
            if i > 0 and (i < self.num_layers-1) and self.skip_connection:
                in_filters = self.g_layers[i] + self.num_skip
            else:
                in_filters = self.g_layers[i]
            
            if self.upsample_mode == 'upsample':
                setattr(self, "conv_uplayer{}".format(i), nn.Sequential(conv(in_filters, self.g_layers[i+1], kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'), 
                                                                    norm_func(self.g_layers[i+1]),
                                                                    nn.LeakyReLU(negative_slope = 0.2),
                                                                    nn.Upsample(scale_factor=2.0, mode='bicubic', align_corners=False)))

            else:
                setattr(self, "conv_uplayer{}".format(i), nn.Sequential(nn.ConvTranspose2d(in_filters, self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias, padding_mode='reflect'), 
                                                                    norm_func(self.g_layers[i+1]),
                                                                    nn.LeakyReLU(negative_slope = 0.2)))
            
            if self.skip_connection:
                #skip_in_filters = self.g_layers[0]
                if self.upsample_mode == 'upsample':
                    setattr(self, "skip_layer{}".format(i), nn.Sequential(conv(self.g_layers[0], self.num_skip, kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'), 
                                                                        norm_func(self.num_skip),
                                                                        nn.LeakyReLU(negative_slope = 0.2),
                                                                        nn.Upsample(scale_factor=2.0, mode='bicubic', align_corners=False)))

                else:
                    setattr(self, "skip_uplayer{}".format(i), nn.Sequential(nn.ConvTranspose2d(self.g_layers[0], self.num_skip, self.filter_size, 2, 1, bias=self.bias, padding_mode='reflect'), 
                                                                        norm_func(self.num_skip),
                                                                        nn.LeakyReLU(negative_slope = 0.2)))


        

    def forward(self, input, cat):
        #print(self.embedding(cat))
        embed = self.embedding(cat).squeeze(1).unsqueeze(2).unsqueeze(3)
        #print(input.size(), embed.size())
        return self.main(input*embed)
