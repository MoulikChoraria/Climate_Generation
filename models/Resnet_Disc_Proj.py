import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.layers import SNConv2d, SNLinear, SNEmbedding


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, downsample=True, n_classes=0, init=True,
                 sn=True, downsample_first=False):
        super().__init__()
        if activation is None or activation == 'None':
            # # define the identity, since we do not add a nonlinear activation.
            activation = lambda x: x
        self.activation = activation
        #self.norm = norm
        self.downsample = downsample
        self.learnable_sc = in_channels != out_channels or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        self.sn = sn
        self.downsample_first = downsample_first and self.downsample

        # # define the two conv layers.
        # self.c1 = nn.Conv2d(in_channels, hidden_channels, ksize, padding=pad)
        # self.c2 = nn.Conv2d(hidden_channels, out_channels, ksize, padding=pad)
        # if self.sn:
        #     nn.utils.spectral_norm(self.c1)
        #     nn.utils.spectral_norm(self.c2)
        if self.sn:
            # # TODO: if this is used in the end, make nice with an if in the beginning!
            self.c1 = SNConv2d(in_channels, hidden_channels, ksize, padding=pad)
            self.c2 = SNConv2d(hidden_channels, out_channels, ksize, padding=pad)
        else:
            self.c1 = nn.Conv2d(in_channels, hidden_channels, ksize, padding=pad)
            self.c2 = nn.Conv2d(hidden_channels, out_channels, ksize, padding=pad)
        if init:
            # # initialization of the conv layers.
            nn.init.zeros_(self.c1.bias)
            nn.init.xavier_uniform_(self.c1.weight, gain=(2 ** 0.5))
            nn.init.xavier_uniform_(self.c2.weight, gain=(2 ** 0.5))
            nn.init.zeros_(self.c2.bias)

        # if self.norm:
        #     m1 = 'For faster discriminator, avoid that (fix forward() otherwise)!'
        #     raise NotImplementedError(m1)
        #     self.b1 = norm(in_channels)
        #     self.b2 = norm(hidden_channels)
        if self.learnable_sc:
            # self.c_sc = nn.Conv2d(in_channels, out_channels, 1, padding=0)
            # if self.sn:
            #     nn.utils.spectral_norm(self.c_sc)
            self.c_sc = SNConv2d(in_channels, out_channels, 1, padding=0)
            if init:
                nn.init.xavier_uniform_(self.c_sc.weight)
                nn.init.zeros_(self.c_sc.bias)

    def forward(self, x, y=None):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        if self.learnable_sc:
            if self.downsample_first:
                x = F.avg_pool2d(x, 2)
            sc = self.c_sc(x)
            if self.downsample and not self.downsample_first:
                sc = F.avg_pool2d(sc, 2)
        else:
            sc = x
        return h + sc




class ResNetProjectionDiscriminator(nn.Module):
    def __init__(self, ch=32, n_classes=5, activation=F.relu, init=True, norm=None,
                 mult_fact=2, ch_input=1, sn=True, downsamples=None, **kwargs):
        super().__init__()
        if downsamples is None:
            # # initialize with the downsampling of resnet_32.
            downsamples = [1, 1, 1, 1, 1, 1]
        self.downsamples = downsamples
        self.activation = activation
        self.init = init
        self.sn = sn
        self.n_classes = n_classes
        # # define a 'macro' for the multiplying factor:
        mf = lambda power: mult_fact ** power
        # # convenience function for the residual block.
        Bl1 = partial(DiscBlock, activation=activation, n_classes=self.n_classes, init=self.init,
                      sn=self.sn)
        self.block1 = Bl1(ch_input, ch, downsample=downsamples[0], hidden_channels=ch, downsample_first=True)
        self.block2 = Bl1(ch, ch * mult_fact, downsample=downsamples[1])
        self.block3 = Bl1(ch * mult_fact, ch * mult_fact, downsample=downsamples[2])
        self.block4 = Bl1(ch * mult_fact, ch * mf(2), downsample=downsamples[3])
        curr = mf(2)

        # # additional blocks to be added.
        for i in range(len(downsamples) - 4):
            setattr(self, 'block{}'.format(i + 5), Bl1(ch * curr, ch * mf(3), 
                                                       downsample=downsamples[4 + i]))
            curr = mf(3)

        # self.l6 = nn.Linear(ch * mf(len(downsamples) - 1), 1, bias=False)
        # if self.sn:
        #     nn.utils.spectral_norm(self.l6)
        n_out = ch*curr
        if self.sn:
            self.l6 = SNLinear(n_out, 1, bias=False)
        else:
            self.l6 = nn.Linear(n_out, 1, bias=False)
        if self.init:
            nn.init.xavier_uniform_(self.l6.weight)

        if n_classes > 0:
            self.l_y = nn.Embedding(n_classes, n_out)
            if self.init:
                nn.init.xavier_uniform_(self.l_y.weight)
            if self.sn:
                nn.utils.spectral_norm(self.l_y)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        for i in range(len(self.downsamples) - 4):
            h = getattr(self, 'block{}'.format(i + 5))(h)
        h = self.activation(h)
        h = h.sum([2, 3])
        output = self.l6(h)
        if y is not None:
            w_y = self.l_y(y)
            output = output + (w_y * h).sum(dim=1, keepdim=True)
        return output