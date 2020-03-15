import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from torchvision import datasets, models, transforms

class MaskGenerator(nn.Module):
    def __init__(self, ngpu=1, nc=3, ngf=32):
        super(MaskGenerator, self).__init__()
        self.ngpu = ngpu
        self.main1 = nn.Sequential(
            # input is (nc) x 224 x 224
            nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 112 x 112
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 56 x 56
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 28 x 28
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 14 x 14
        )
        self.main2 = nn.Sequential(
            # input is (nc) x 224 x 224
            nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 112 x 112
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 56 x 56
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 28 x 28
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 14 x 14
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 28 x 28
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 56 x 56
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 112 x 112
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(nc*2),
            # nn.ReLU(True),
            # state size. (ngf) x 224 x 224
            nn.Sigmoid()
        )

    def forward(self, source, target):
        f1 = self.main1(source) 
        f2 = self.main2(target)
        f3 = torch.cat((f1, f2), 1)
        mask = self.deconv(f3)
        syn = (source * mask * 2) 
        # syn[syn > 1] = 1.0
        # syn[syn < 0] = 0
        return syn

class Discriminator(nn.Module):
    def __init__(self, ngpu=1, nc=3, ndf=32):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
             # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class IDT_Net(nn.Module):
    def __init__(self, ngpu=1, nc=3, ndf=32, ngf=32):
        super(IDT_Net, self).__init__()
        self.ngpu = ngpu
        self.MaskGenerator = MaskGenerator(ngpu=1, nc=3, ngf=16)
        self.Discriminator = Discriminator(ngpu=1, nc=3, ndf=4)
        self.Backbone = models.resnet18(pretrained=True)

    def forward(self, source, target):
        new_source = self.MaskGenerator(source)
        new_source_logits = self.Discriminator(new_source)
        target_logits = self.Discriminator(target)

        source_feature = self.Backbone(source)
        new_source_feature = self.Backbone(new_source)

        return source_feature, new_source_feature, target_logits, new_source_logits, new_source
