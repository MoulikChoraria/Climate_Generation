import torch.nn as nn
import torch.nn.functional as F
from models.layers import CategoricalConditionalBatchNorm
from functools import partial
from torch import tanh


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, n_classes=0, norm=None, init=True):
        super().__init__()
        if activation is None or activation == 'None':
            # # define the identity, since we do not add a nonlinear activation.
            activation = lambda x: x
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes

        # # define the two conv layers.
        self.c1 = nn.Conv2d(in_channels, hidden_channels, ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, ksize, padding=pad)

        if norm is None:
            norm = CategoricalConditionalBatchNorm if n_classes > 0 else nn.BatchNorm2d
        if n_classes == 0:
            self.b1 = norm(in_channels)
            self.b2 = norm(hidden_channels)
        else:
            self.b1 = norm(in_channels, n_classes)
            self.b2 = norm(hidden_channels, n_classes)
        if init:
            # # initialization of the conv layers.
            nn.init.zeros_(self.c1.bias)
            nn.init.xavier_uniform_(self.c1.weight, gain=(2 ** 0.5))
            nn.init.xavier_uniform_(self.c2.weight, gain=(2 ** 0.5))
            nn.init.zeros_(self.c2.bias)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, 1, padding=0)
            if init:
                nn.init.xavier_uniform_(self.c_sc.weight)
                nn.init.zeros_(self.c_sc.bias)

    def forward(self, x, y=None):
        h = x
        h = self.b1(h, y) if y is not None else self.b1(h)
        h = self.activation(h)
        if self.upsample:
            h = F.upsample(h, scale_factor=2)
        h = self.c1(h)
        h = self.b2(h, y) if y is not None else self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.learnable_sc:
            if self.upsample:
                x = F.upsample(x, scale_factor=2)
            sc = self.c_sc(x)
        else:
            sc = x
        return h + sc


#from utils.random_samples import sample_continuous, sample_categorical


def return_norm(type_norm, n_classes=0):
    """
    Returns the normalization function to use.
    :param type_norm: int or str; The type of normalization.
    :param n_classes: int, the number of classes.
    :return: The batch normalization function to call.
    """
    if type_norm is None:
        # # return as a function of functions, i.e. the first time
        # # to initialize the 'normalization' and the second to
        # # call it during the forward() step.
        return lambda *args, **kwargs: (lambda y: y)
    if n_classes == 0:
        if type_norm == 0 or type_norm == 'batch':
            return nn.BatchNorm2d
        elif type_norm == 1 or type_norm == 'instance':
            return nn.InstanceNorm2d
    else:
        if type_norm == 0 or type_norm == 'batch':
            return CategoricalConditionalBatchNorm
        else:
            raise RuntimeError('Not implemented the class-conditional batch norms!')


class ResNetGenerator(nn.Module):
    def __init__(self, ch=32, dim_z=128, bottom_width=2, activation=F.relu,
                 n_classes=5, use_out_activ=True, mult_fact=2, z_mult=None,
                 add_blocks=3, init=True, type_norm='batch', out_ch=1,
                 distribution='normal', training=True, out_dim=64):
        """
        Resnet generator; based on SNGAN but augmented with new functionalities.
        :param ch: int, base channels for each residual block.
        :param dim_z: int, dimension of the noise.
        :param bottom_width: int, dimension to start the upsampling.
        :param activation: torch function or None. The activation for each layer.
        :param n_classes: int, the number of classes if conditional generator.
        :param use_out_activ: bool. If True, use the typical tanh in the output.
        :param mult_fact: int. If > 1, then increase the channels in each
            block by mult_fact in the respective power.
        :param z_mult: int or None. If int, insert that many global
            transformations before starting the convolutions.
        :param add_blocks: Int. If  > 0, then we add new residual blocks
            in the output.
        :param type_norm: int/str. The type of normalization for each convolution.
        :param out_ch: int, the number of output channels.
        :param distribution: str, the name of the distribution to sample from.
        :param training: bool, if True it indicates it is in training mode.
        """
        super().__init__()
        self.bottom_width = bottom_width
        self.num_upsamples = 0
        width = out_dim
        while width>bottom_width:
            self.num_upsamples += 1
            width = width//2
        assert self.num_upsamples >= 3, "expected output too small, reconfigure network"

        self.out_activ = tanh if use_out_activ else lambda x: x
        if activation is None or activation == 'None':
            # # define the identity, since we do not add a nonlinear one.
            activation = lambda x: x
        self.activation = activation
        self.dim_z = dim_z
        self.input_dim = dim_z
        self.n_classes = n_classes
        self.distribution = distribution
        # # define a 'macro' for the multiplying factor:
        mf = lambda power: mult_fact ** power
        self.z_mult = z_mult
        self.add_blocks = add_blocks
        self.training = training
        self.init = init
        self.norm = norm = return_norm(type_norm, n_classes=self.n_classes)

        # # convenience function for the residual block.
        Bl1 = partial(Block, activation=activation, n_classes=self.n_classes, init=self.init,
                      norm=norm)
        # # Global affine transformations of z.
        if z_mult is not None and z_mult > 0:
            for l in range(1, z_mult + 1):
                setattr(self, 'zgL{}'.format(l), nn.Linear(self.dim_z, self.dim_z))
        # # define the (main) layers.
        self.l1 = nn.Linear(dim_z, (bottom_width ** 2) * ch * mf(4))
        if self.init:
            nn.init.xavier_uniform_(self.l1.weight)
            nn.init.zeros_(self.l1.bias)
        self.block2 = Bl1(ch * mf(4), ch * mf(4), upsample=True)
        self.block3 = Bl1(ch * mf(4), ch * mf(3), upsample=True)
        self.block4 = Bl1(ch * mf(3), ch * mf(3), upsample=True)
        num_upsamples = self.num_upsamples-3
        #self.block4 = Bl1(ch * mf(3), ch * mf(2), upsample=False)
        curr = ch * mf(3)
        ly = 3
        next_ch = ch * mf(ly)

        add_blocks = max(self.add_blocks, num_upsamples)
        # # add residual blocks if requested.
        for i in range(add_blocks):
            up = True  if num_upsamples>0 else False
            setattr(self, 'block{}'.format(i + 5), Bl1(curr, next_ch, upsample=up))
            num_upsamples-=1
            curr = next_ch
            if ly > 2:
                next_ch = ch * mf(ly-1)
                ly -= 1
            
        #print(curr, ch * mf(1))
        if self.n_classes == 0:
            self.b5 = self.norm(curr)
        else:
            self.b5 = self.norm(curr, n_classes)

        self.l5 = nn.Conv2d(curr, out_ch, kernel_size=3, stride=1, padding=1)
        if self.init:
            nn.init.xavier_uniform_(self.l5.weight)
            nn.init.zeros_(self.l5.bias)

    def forward(self, z, y):
        anyparam = next(self.parameters())
        #if z is None:
        #    z = sample_continuous(self.dim_z, anyparam, batchsize=batchsize, distribution=self.distribution)
        #if y is None and self.n_classes > 0:
        #    y = sample_categorical(self.n_classes, anyparam, batchsize=batchsize, distribution='uniform')
        #if (y is not None) and z.shape[0] != y.shape[0]:
        #    m1 = 'z.shape[0] != y.shape[0], z.shape[0]={}, y.shape[0]={}'
        #    raise Exception(m1.format(z.shape[0], y.shape[0]))
        # # apply the global transformations if they exist.
        #if self.z_mult is not None and self.z_mult > 0:
        #    for l in range(1, self.z_mult + 1):
        #        z = self.activation(getattr(self, 'zgL{}'.format(l))(z))
        h = z
        h = h.reshape(h.shape[0], -1)
        #print(h.size())
        h = self.l1(h)
        h = h.reshape(h.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.block4(h, y)
        for i in range(self.add_blocks):
            h = getattr(self, 'block{}'.format(i + 5))(h, y)
        #print(h.size())
        if self.n_classes > 0:
            h = self.b5(h, y)
        else:
            h = self.b5(h)
        h = self.activation(h)
        h = self.out_activ(self.l5(h))
        return h