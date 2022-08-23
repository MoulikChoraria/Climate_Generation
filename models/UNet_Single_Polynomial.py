import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN
from numpy import sqrt
import torch
from torch.nn.functional import linear  


class UNet_1Poly(nn.Module):
    def __init__(self, g_layers=[], activation_fn=True, inject_z=True, transform_rep=6, \
                                                    transform_z=False, concat_injection=False, norm='instance', filter_size = 4, bias = False):
        super(UNet_1Poly, self).__init__()
        
        """Symmetric upsampling and downsampling, first and last layer retain same input and output last two dimensions (H X W)"""

        self.downsampling_layers = int((len(g_layers)-3)/2)
        self.activation_fn = activation_fn
        self.filter_size = filter_size
        #self.downsample_filter = 3
        self.g_layers = g_layers
        self.inject_z = inject_z
        self.num_layers = len(self.g_layers) - 1  # minus the input/output sizes 
        self.transform_rep = transform_rep
        self.transform_z = transform_z
        self.concat_injection = concat_injection
        self.norm = norm
        self.bias = bias

        if self.transform_z:
            for i in range(self.transform_rep):
                setattr(self, "global{}".format(i), nn.Sequential(
                                                        nn.Linear(self.g_layers[0], self.g_layers[0]),
                                                        nn.ReLU()))

        for i in range(self.num_layers):
                if i < (self.downsampling_layers+1) and i > 0:
                    if self.inject_z:
                    ### inject layers always use instance_norm
                        if not self.activation_fn:
                            setattr(self, "inject{}".format(i), nn.Sequential(
                                                                    nn.Conv2d(self.g_layers[0], self.g_layers[i+1], self.filter_size, 1, 1, bias=self.bias, padding_mode='reflect'),
                                                                    nn.InstanceNorm2d(self.g_layers[i+1]),
                                                                    nn.AvgPool2d(kernel_size = (2**i+1), stride = 2**i, padding = 1)))
                        else:
                            setattr(self, "inject{}".format(i), nn.Sequential(
                                                                    nn.Conv2d(self.g_layers[0], self.g_layers[i+1], self.filter_size, 1, 1, bias=self.bias, padding_mode='reflect'), 
                                                                    nn.InstanceNorm2d(self.g_layers[i+1]),
                                                                    nn.AvgPool2d(kernel_size = (2**i+1), stride = 2**i, padding = 1),
                                                                    nn.ReLU()))

                    ### main downsampling via strides and not pooling, fixed factor = 2
                    if not self.activation_fn:
                        if self.norm == 'instance':
                            setattr(self, "layer{}".format(i), nn.Sequential(
                                                                    nn.Conv2d(self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias, padding_mode='reflect'),
                                                                    nn.InstanceNorm2d(self.g_layers[i+1])))  
                        else:
                            setattr(self, "layer{}".format(i), nn.Sequential(
                                                                    nn.Conv2d(self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias, padding_mode='reflect'),
                                                                    nn.BatchNorm2d(self.g_layers[i+1])))                              
                    else:
                        if self.norm == 'instance':
                            setattr(self, "layer{}".format(i), nn.Sequential(
                                                                    nn.Conv2d(self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias, padding_mode='reflect'), 
                                                                    nn.InstanceNorm2d(self.g_layers[i+1]),
                                                                    nn.ReLU()))
                        else:
                            setattr(self, "layer{}".format(i), nn.Sequential(
                                                                    nn.Conv2d(self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias, padding_mode='reflect'), 
                                                                    nn.BatchNorm2d(self.g_layers[i+1]),
                                                                    nn.ReLU()))
                
                elif i == 0:
                ### fixed filter size = 3, no downsampling    
                    if not self.activation_fn:
                        if self.norm == 'instance':
                            setattr(self, "layer{}".format(i), nn.Sequential(
                                                                    nn.Conv2d(self.g_layers[i], self.g_layers[i+1], 3, 1, 1, bias=self.bias, padding_mode='reflect'),
                                                                    nn.InstanceNorm2d(self.g_layers[i+1])))  
                        else:
                            setattr(self, "layer{}".format(i), nn.Sequential(
                                                                    nn.Conv2d(self.g_layers[i], self.g_layers[i+1], 3, 1, 1, bias=self.bias, padding_mode='reflect'),
                                                                    nn.BatchNorm2d(self.g_layers[i+1])))                              
                    else:
                        if self.norm == 'instance':
                            setattr(self, "layer{}".format(i), nn.Sequential(
                                                                    nn.Conv2d(self.g_layers[i], self.g_layers[i+1], 3, 1, 1, bias=self.bias, padding_mode='reflect'), 
                                                                    nn.InstanceNorm2d(self.g_layers[i+1]),
                                                                    nn.ReLU()))
                        else:
                            setattr(self, "layer{}".format(i), nn.Sequential(
                                                                    nn.Conv2d(self.g_layers[i], self.g_layers[i+1], 3, 1, 1, bias=self.bias, padding_mode='reflect'), 
                                                                    nn.BatchNorm2d(self.g_layers[i+1]),
                                                                    nn.ReLU()))

                elif i == self.num_layers - 1:
                ### fixed filter size = 3, no upsampling    
                    if not self.activation_fn:
                        setattr(self, "layer{}".format(i), nn.Sequential(
                                                                nn.Conv2d(self.g_layers[i], self.g_layers[i+1], 3, 1, 1, bias=self.bias, padding_mode='reflect'),
                                                                nn.Sigmoid()))
                    else:
                        setattr(self, "layer{}".format(i), nn.Sequential(
                                                                nn.Conv2d(self.g_layers[i], self.g_layers[i+1], 3, 1, 1, bias=self.bias, padding_mode='reflect'),
                                                                nn.Sigmoid())) 
                else:
                ### upsampling via convs/pooling preferred to convtranspose   
                    if self.inject_z:
                        ### inject layers from input, always need to be upsampled
                        if i < 2*self.downsampling_layers:
                            if not self.activation_fn:
                                setattr(self, "inject{}".format(i), nn.Sequential(
                                                                        nn.Conv2d(self.g_layers[0], self.g_layers[i+1], self.filter_size, 1, 1, bias=self.bias, padding_mode='reflect'),
                                                                        nn.InstanceNorm2d(self.g_layers[i+1]),
                                                                        nn.AvgPool2d(kernel_size = (2**(2*self.downsampling_layers - i)+1), stride = 2**(2*self.downsampling_layers - i), padding = 1)))
                            else:
                                setattr(self, "inject{}".format(i), nn.Sequential(
                                                                        nn.Conv2d(self.g_layers[0], self.g_layers[i+1], self.filter_size, 1, 1, bias=self.bias, padding_mode='reflect'),
                                                                        nn.InstanceNorm2d(self.g_layers[i+1]),
                                                                        nn.AvgPool2d(kernel_size = (2**(2*self.downsampling_layers - i)+1), stride = 2**(2*self.downsampling_layers - i), padding = 1),
                                                                        nn.ReLU()))

                        else:
                            if not self.activation_fn:
                                setattr(self, "inject{}".format(i), nn.Sequential(
                                                                        nn.Conv2d(self.g_layers[0], self.g_layers[i+1], 3, 1, 1, bias=self.bias, padding_mode='reflect'),
                                                                        nn.InstanceNorm2d(self.g_layers[i+1])))
                            else:
                                setattr(self, "inject{}".format(i), nn.Sequential(
                                                                        nn.Conv2d(self.g_layers[0], self.g_layers[i+1], 3, 1, 1, bias=self.bias, padding_mode='reflect'),
                                                                        nn.InstanceNorm2d(self.g_layers[i+1]),
                                                                        nn.ReLU()))

                    if not self.activation_fn:
                        if self.norm == 'instance':
                            setattr(self, "layer{}".format(i), nn.Sequential(
                                                                    nn.ConvTranspose2d(self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias),
                                                                    nn.InstanceNorm2d(self.g_layers[i+1])))
                        else:
                            setattr(self, "layer{}".format(i), nn.Sequential(
                                                                    nn.ConvTranspose2d(self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias),
                                                                    nn.BatchNorm2d(self.g_layers[i+1])))                            
                        if self.concat_injection:
                            if self.norm == 'instance':
                                setattr(self, "layer{}".format(i), nn.Sequential(
                                                                    nn.ConvTranspose2d(2*self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias),
                                                                    nn.InstanceNorm2d(self.g_layers[i+1])))
                            else:
                                setattr(self, "layer{}".format(i), nn.Sequential(
                                                                    nn.ConvTranspose2d(2*self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias),
                                                                    nn.BatchNorm2d(self.g_layers[i+1])))                                
                    else:
                        if self.norm == 'instance':
                            setattr(self, "layer{}".format(i), nn.Sequential(
                                                                    nn.ConvTranspose2d(self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias), 
                                                                    nn.InstanceNorm2d(self.g_layers[i+1]),
                                                                    nn.ReLU()))
                        else:
                            setattr(self, "layer{}".format(i), nn.Sequential(
                                                                    nn.ConvTranspose2d(self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias), 
                                                                    nn.BatchNorm2d(self.g_layers[i+1]),
                                                                    nn.ReLU()))                            
                        if self.concat_injection:
                            if self.norm == 'instance':
                                setattr(self, "layer{}".format(i), nn.Sequential(
                                                                        nn.ConvTranspose2d(2*self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias), 
                                                                        nn.InstanceNorm2d(self.g_layers[i+1]),
                                                                        nn.ReLU())) 
                            else:
                                setattr(self, "layer{}".format(i), nn.Sequential(
                                                                        nn.ConvTranspose2d(2*self.g_layers[i], self.g_layers[i+1], self.filter_size, 2, 1, bias=self.bias), 
                                                                        nn.BatchNorm2d(self.g_layers[i+1]),
                                                                        nn.ReLU()))                                   

    def forward(self, x):
        z = x.squeeze(3).squeeze(2)
        #z = x.
        #if self.transform_z:
        #    for i in range(self.transform_rep):
        #        z = getattr(self, "global{}".format(i))(z)
        #x = z.unsqueeze(2).unsqueeze(3)
        for i in range(self.num_layers):
            #print("x",x.size())
            x = getattr(self, "layer{}".format(i))(x)
            #print("x",x.size())

            if self.inject_z:
                if i < self.num_layers - 1 and i > 0:
                    #a = getattr(self, "inject{}".format(i))(z)
                    #a = a.unsqueeze(2).unsqueeze(3).expand(x.size(0), x.size(1), x.size(2), x.size(3))
                    
                    a = getattr(self, "inject{}".format(i))(z)
                    #print("a", a.size())
                    #x = getattr(self, "layer{}".format(i))(x)
                    #a = a.unsqueeze(2).unsqueeze(3).expand(x.size(0), x.size(1), x.size(2), x.size(3))
                    if self.concat_injection:
                        x = torch.cat((x, a), dim=1)
                    else:
                        x *= a 
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
