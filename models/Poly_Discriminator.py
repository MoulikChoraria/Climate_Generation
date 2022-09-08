import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN
from numpy import sqrt
import torch
from torch.nn.functional import linear
import numpy as np 

class printshape(nn.Module):
  def __init__(self):
        super(printshape, self).__init__()
  
  def forward(self, x):
    print(x.size())
    return x

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


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero'):

    padder = None
    ### assuming stride=1, dilation=1
    #### H_in+2×padding[0]−dilation[0]×(kernel_size[0]−1)

    to_pad = (kernel_size - 1)
    if to_pad==0:
        padder = None 
    elif to_pad==1:
        if pad == 'reflect':
                #### format = (left, right, top, bottom)
                padder = nn.ReflectionPad2d((0, 1, 0, 1))
        else:
            ### zero pad
            padder = nn.ZeroPad2d((0, 1, 0, 1))
    elif to_pad > 1:
        if to_pad % 2 == 0:
            if pad == 'reflect':
                #### format = (left, right, top, bottom)
                padder = nn.ReflectionPad2d((to_pad//2, to_pad//2, to_pad//2, to_pad//2))
            else:
                ### zero pad
                padder = nn.ZeroPad2d((to_pad//2, to_pad//2, to_pad//2, to_pad//2))
            
        else:
            if pad == 'reflect':
                #### format = (left, right, top, bottom)
                padder = nn.ReflectionPad2d((to_pad//2, to_pad//2+1, to_pad//2, to_pad//2+1))
            else:
                ### zero pad
                padder = nn.ZeroPad2d((to_pad//2, to_pad//2+1, to_pad//2, to_pad//2+1))
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=0, bias=bias)
    #printme = printshape()



    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)

def conv_down(in_f, out_f, kernel_size, bias=True, pad='zero', down_mode='pool', pool_mode=None, pool_kernel_size=-1):
    ### every reduction by atmost a factor of 2

    if(pad == 'zero'):
        pad_func = nn.ZeroPad2d
    else:
        pad_func = nn.RelectionPad2d

    padder1 = None
    padder2 = None
    pooler = None
    ### assuming dilation=1
    ### conv
    #### H_out =⌊(H_in +2×padding[0]−dilation[0]×(kernel_size[0]−1)−1)/stride[0] +1⌋


    if(down_mode == 'pool'):
        stride = 1
        padder1_pad = (kernel_size - stride)/stride
        if(padder1_pad > 0):
            if padder1_pad % 2 == 0:
                padder1 = pad_func((padder1_pad//2, padder1_pad//2, padder1_pad//2, padder1_pad//2))
            else:
                padder1 = pad_func((padder1_pad//2, padder1_pad//2+1, padder1_pad//2, padder1_pad//2+1))

        pool_stride = 2
        ### pool
        #### H_out =⌊(H_in +2×padding[0]−kernel_size[0])/scale +1⌋
        padder2_pad = (pool_kernel_size-pool_stride)
        if(padder2_pad > 0):
            if padder1_pad % 2 == 0:
                padder2 = pad_func((padder2_pad//2, padder2_pad//2, padder2_pad//2, padder2_pad//2))
            else:
                padder2 = pad_func((padder2_pad//2, padder2_pad//2+1, padder2_pad//2, padder2_pad//2+1))
        
        if(pool_mode == 'avg'):
            pool_func = nn.AvgPool2D
        else:
            pool_func = nn.MaxPool2D

        pooler = pool_func(pool_kernel_size, pool_stride)

    else:
        stride=2
        padder1_pad = np.ceil(kernel_size/stride - 1)

        if(padder1_pad > 0):
            if padder1_pad % 2 == 0:
                padder1 = pad_func((padder1_pad//2, padder1_pad//2, padder1_pad//2, padder1_pad//2))
            else:
                padder1 = pad_func((padder1_pad//2, padder1_pad//2+1, padder1_pad//2, padder1_pad//2+1))
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=0, bias=bias)
    #printme = printshape()

    layers = filter(lambda x: x is not None, [padder1, convolver, padder2, pooler])
    return nn.Sequential(*layers)


def conv_transpose(in_f, out_f, kernel_size, stride=2, bias=True, pad='zero'):
    ### assume scale_up = stride
    ### H_out =(H_in−1)×stride[0]−2×padding[0]+dilation×(kernel_size[0]−1)+output_padding[0]+1
    ### dilation is fixed = 1

    ### calculate input padding such that each output_dim = stride*input_dim

    to_pad = kernel_size - stride
    padder = None

    if to_pad==0:
        padder = None 
    elif to_pad > 0:
        if to_pad % 2 == 0:
            to_pad = to_pad//2
        else:
            to_pad = (to_pad+1)//2
            padder = nn.ZeroPad2d((0, 1, 0, 1))
    else:
        print("Need Dilation/Output padding")
        raise NotImplementedError
  
    convolver = nn.ConvTranspose2d(in_f, out_f, kernel_size, stride, padding=(to_pad, to_pad), bias=bias, padding_mode='zeros')
    #printme = printshape()


    layers = filter(lambda x: x is not None, [convolver, padder])
    return nn.Sequential(*layers)



class conditional_polydisc(nn.Module):
    def __init__(self, input_dim, num_classes=5, d_layers=[], remove_hot = 0, inject_z=True, transform_rep=1, \
                    #transform_z=False, 
                            norm='instance', filter_size = 3, bias = False,\
                            skip_connection = False, num_skip=4, skip_size=1, residual=False, downsample_mode = 'pooling', pool_type='avg', pool_filter = 2):
        super(conditional_polydisc, self).__init__()

        self.residual = residual
        self.num_layers = len(d_layers)-1
        self.allowed_injections = (self.num_layers - 1) - remove_hot
        self.filter_size = filter_size
        #self.downsample_filter = 3
        self.inject_z = inject_z
        self.num_layers = len(self.g_layers) - 1  # minus the input/output sizes 
        #self.transform_z = transform_z
        self.norm = norm
        self.bias = bias
        self.skip_connection = skip_connection
        self.num_skip = num_skip
        self.skip_size = skip_size
        self.d_layers = d_layers
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.transform_rep = transform_rep
        self.down_mode = downsample_mode
        self.pool_filter_size = pool_filter
        self.pool_filter_type = pool_type
        
        self.embedding = nn.Embedding(self.num_classes, self.input_dim)
        
        # if self.transform_z:
        #     for i in range(self.transform_rep):
        #         setattr(self, "global{}".format(i), nn.Sequential(
        #                                                 nn.Linear(self.g_layers[0], self.g_layers[0]),
        #                                                 nn.ReLU()))
        
        total_injections = 0
        track_dim = self.input_dim
        
        if(self.norm == 'instance'):
            norm_func = nn.InstanceNorm2d
        else:
            norm_func = nn.BatchNorm2d


        for i in range(self.num_layers):
            if i > 0 and (i < self.num_layers-1) and self.skip_connection:
                in_filters = self.g_layers[i] + self.num_skip
            else:
                in_filters = self.g_layers[i]
            
            if (i < self.num_layers-1):
                setattr(self, "conv_layer{}".format(i), nn.Sequential(conv_down(in_filters, self.g_layers[i+1], kernel_size = self.filter_size, bias=self.bias, pad='reflect', \
                                                                            down_mode = self.down_mode, pool_mode=self.pool_filter_type, pool_kernel_size=self.pool_filter_size), 
                                                                    norm_func(self.g_layers[i+1]),
                                                                    nn.LeakyReLU(negative_slope = 0.2),
                                                                    conv(self.g_layers[i+1], self.g_layers[i+1], kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'), 
                                                                    norm_func(self.g_layers[i+1]),
                                                                    nn.LeakyReLU(negative_slope = 0.2)))
                track_dim = (track_dim[0]//2, track_dim[1]//2)

            else:
                lin_dim = final_layer_filters*(track_dim[0]//2)*(track_dim[1]//2)
                final_layer_filters = 16
                if self.skip_connection:
                        curr_layer_filters =  self.g_layers[i] + self.num_skip
                        #final_layer_filters = 128 + self.num_skip
                else:
                    curr_layer_filters =  self.g_layers[i]

                setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                            conv_down(curr_layer_filters, final_layer_filters, kernel_size = self.filter_size, bias=self.bias, pad='reflect', \
                                                                            down_mode = self.down_mode, pool_mode=self.pool_filter_type, pool_kernel_size=self.pool_filter_size), 
                                                            norm_func(final_layer_filters),
                                                            nn.LeakyReLU(negative_slope = 0.2)))

                setattr(self, "linear_layer{}".format(i+1), nn.Sequential(
                                                nn.Linear(lin_dim, lin_dim//8, bias=self.bias),
                                                nn.ReLU(),
                                                nn.Linear(lin_dim, self.g_layers[i+1], bias=self.bias)))
                                                
                
            if self.skip_connection and i < self.num_layers-1:
                setattr(self, "skip_layer{}".format(i), nn.Sequential(conv(self.g_layers[0], self.num_skip, kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'), 
                                                                    norm_func(self.num_skip),
                                                                    nn.AdaptiveAvgPool2d(track_dim),
                                                                    nn.LeakyReLU(negative_slope = 0.2)))

            if self.inject_z and total_injections < self.allowed_injections and i < self.num_layers-1:
                #skip_in_filters = self.g_layers[0]
                total_injections += 1
                setattr(self, "skip_layer{}".format(i), nn.Sequential(conv(self.g_layers[0], self.self.g_layers[i+1], kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'), 
                                                                norm_func(self.num_skip),
                                                                nn.AdaptiveAvgPool2d(track_dim),
                                                                nn.LeakyReLU(negative_slope = 0.2)))


        

    def forward(self, input, cat):
        #print(self.embedding(cat))
        embed = self.label_embedding(cat)
        reshape_embed = embed.reshape(input.size(0), 1, input.size(2), input.size(3))
        z = torch.cat((input, reshape_embed), dim=1)
        
        # combined_input = input
        # return (self.main(combined_input).squeeze(3).squeeze(2))
        # if(self.transform_z):
        #     input = getattr(self, "global_transform")(input)

        #z = input*embed
        injections = self.allowed_injections

        for i in range(self.num_layers):
            #print("x",x.size())
            if(i == 0):
                x = z

            x = getattr(self, "conv_layer{}".format(i))(x)
            print("conv", i, x.size())

            if(self.inject_z and injections>0):
                if i < self.num_layers - 1:

                    injections -= 1
                    a = getattr(self, "inject_layer{}".format(i))(z)
                    print("inject", i, a.size())

                    if not self.residual:
                        x *= a
                    else: 
                    ### residual
                        x = x*a + x
            
            if self.skip_connection:
                if i < self.num_layers - 1:
                    skip = getattr(self, "skip_layer{}".format(i))(z)
                    print("skip", i, skip.size())
                    print("out", i, x.size())
                    x = torch.cat((x, skip), dim=1)

            #apply injection

        x = getattr(self, "linear_layer{}".format(self.num_layers))(x)

        return x