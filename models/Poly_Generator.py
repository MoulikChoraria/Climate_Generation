import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN
from numpy import sqrt
import torch
from torch.nn.functional import linear 

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



class conditional_polygen(nn.Module):
    def __init__(self, input_dim, num_classes=5, layers=[], remove_hot = 0, inject_z=True, transform_rep=1, \
                    transform_z=False, norm_type='instance', filter_size = 3, bias = False,\
                            skip_connection = False, num_skip=4, skip_size=1, residual=False, up_mode = 'upsample'):
        super(conditional_polygen, self).__init__()

        self.residual = residual
        self.num_layers = len(layers)-1
        self.allowed_injections = (self.num_layers - 1) - remove_hot
        self.filter_size = filter_size

        #self.downsample_filter = 3
        self.layers = layers
        self.inject_z = inject_z
        self.num_layers = len(self.layers) - 1  # minus the input/output sizes 
        self.transform_z = transform_z
        self.norm_type = norm_type
        self.bias = bias
        self.skip_connection = skip_connection
        self.num_skip = num_skip
        self.skip_size = skip_size
        self.layers = layers
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.transform_rep = transform_rep
        self.upsample_mode = up_mode
        self.embedding = nn.Embedding(self.num_classes, self.input_dim)
        
        if self.transform_z:
            for i in range(self.transform_rep):
                setattr(self, "global{}".format(i), nn.Sequential(
                                                        nn.Linear(self.layers[0], self.layers[0]),
                                                        nn.ReLU()))
        
        total_injections = 0
        
        if(self.norm_type == 'instance'):
            norm_func = nn.InstanceNorm2d
        else:
            norm_func = nn.BatchNorm2d


        for i in range(self.num_layers):
            if i > 0 and (i < self.num_layers-1) and self.skip_connection:
                in_filters = self.layers[i] + self.num_skip
            else:
                in_filters = self.layers[i]
            
            if (i < self.num_layers-1):
                if self.upsample_mode == 'upsample':
                    setattr(self, "conv_layer{}".format(i), nn.Sequential(conv(in_filters, self.layers[i+1], kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'), 
                                                                        norm_func(self.layers[i+1]),
                                                                        nn.LeakyReLU(negative_slope = 0.2),
                                                                        nn.Upsample(scale_factor=2.0, mode='bicubic', align_corners=False),
                                                                        conv(self.layers[i+1], self.layers[i+1], kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'), 
                                                                        norm_func(self.layers[i+1]),
                                                                        nn.LeakyReLU(negative_slope = 0.2)))

                else:
                    setattr(self, "conv_layer{}".format(i), nn.Sequential(conv_transpose(in_filters, self.layers[i+1], kernel_size = self.filter_size, stride = 2, bias=self.bias, pad='reflect'), 
                                                                        norm_func(self.layers[i+1]),
                                                                        nn.LeakyReLU(negative_slope = 0.2),
                                                                        conv(self.layers[i+1], self.layers[i+1], kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'), 
                                                                        norm_func(self.layers[i+1]),
                                                                        nn.LeakyReLU(negative_slope = 0.2)))

            else:
                if self.skip_connection:
                        curr_layer_filters =  self.layers[i] + self.num_skip
                        #final_layer_filters = 128 + self.num_skip
                        final_layer_filters = 128
                else:
                    curr_layer_filters =  self.layers[i]
                    final_layer_filters = 128

                setattr(self, "conv_layer{}".format(i), nn.Sequential(
                                                            conv(curr_layer_filters, 128, kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'),
                                                            norm_func(128),
                                                            nn.LeakyReLU(negative_slope = 0.2)))

                setattr(self, "conv_layer{}".format(i+1), nn.Sequential(
                                                conv(final_layer_filters, 128, kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'),
                                                norm_func(128),
                                                nn.LeakyReLU(negative_slope = 0.2),
                                                conv(128, self.layers[i+1], kernel_size = 1, stride = 1, bias=self.bias, pad='reflect')))#,
                                                #nn.Sigmoid()))
                                                
                
            if self.skip_connection and i < self.num_layers-1:
                #skip_in_filters = self.layers[0]
                if self.upsample_mode == 'upsample' or i==0:
                ### upsample purely via upsampling module
                    setattr(self, "skip_layer{}".format(i), nn.Sequential(conv(self.layers[0], self.num_skip, kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'), 
                                                                        norm_func(self.num_skip),
                                                                        nn.LeakyReLU(negative_slope = 0.2),
                                                                        nn.Upsample(scale_factor=2**(i+1), mode='bicubic', align_corners=False)))

                else:
                ### upsample via convtranspose2d + upsample_module    
                    setattr(self, "skip_layer{}".format(i), nn.Sequential(conv_transpose(self.layers[0], self.num_skip, kernel_size = self.filter_size, stride = 2, bias=self.bias, pad='reflect'), 
                                                                        norm_func(self.num_skip),
                                                                        nn.LeakyReLU(negative_slope = 0.2),
                                                                        nn.Upsample(scale_factor=2**(i), mode='bicubic', align_corners=False)))

            if self.inject_z and total_injections < self.allowed_injections:
                #skip_in_filters = self.layers[0]
                total_injections += 1
                if self.upsample_mode == 'upsample' or i==0:
                ### upsample purely via upsampling module
                    setattr(self, "inject_layer{}".format(i), nn.Sequential(conv(self.layers[0], self.layers[i+1], kernel_size = self.filter_size, stride = 1, bias=self.bias, pad='reflect'), 
                                                                        norm_func(self.layers[i+1]),
                                                                        nn.LeakyReLU(negative_slope = 0.2),
                                                                        nn.Upsample(scale_factor=2**(i+1), mode='bicubic', align_corners=False)))

                else:
                ### upsample via convtranspose2d + upsample_module    
                    setattr(self, "inject_layer{}".format(i), nn.Sequential(conv_transpose(self.layers[0], self.layers[i+1], kernel_size = self.filter_size, stride = 2, bias=self.bias, pad='reflect'), 
                                                                        norm_func(self.layers[i+1]),
                                                                        nn.LeakyReLU(negative_slope = 0.2),
                                                                        nn.Upsample(scale_factor=2**(i), mode='bicubic', align_corners=False)))


        

    def forward(self, input, cat):
        #print(self.embedding(cat))
        embed = self.embedding(cat).squeeze(1).unsqueeze(2).unsqueeze(3)
        #print(input.size(), embed.size())
        if(self.transform_z):
            input = getattr(self, "global_transform")(input)

        z = input*embed
        s_tuple = z.size()
        ### shape = (batch, input_dim, 1, 1)
        z = z.view((s_tuple[0], s_tuple[1]//16, 4, 4))
        ### shape = (batch, input_dim//16, 4, 4)
        injections = self.allowed_injections

        for i in range(self.num_layers):
            #print("x",x.size())
            if(i == 0):
                x = z

            x = getattr(self, "conv_layer{}".format(i))(x)
            #print("conv", i, x.size())

            if(self.inject_z and injections>0):
                if i < self.num_layers - 1:

                    injections -= 1
                    a = getattr(self, "inject_layer{}".format(i))(z)
                    #print("inject", i, a.size())

                    if not self.residual:
                        x *= a
                    else: 
                    ### residual
                        x = x*a + x
            
            if self.skip_connection:
                if i < self.num_layers - 1:
                    skip = getattr(self, "skip_layer{}".format(i))(z)
                    #print("skip", i, skip.size())
                    #print("out", i, x.size())
                    x = torch.cat((x, skip), dim=1)

            #apply injection

        x = getattr(self, "conv_layer{}".format(self.num_layers))(x)

        return x