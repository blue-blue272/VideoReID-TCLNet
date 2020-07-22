from __future__ import absolute_import

import torch
import math
import copy
import torchvision
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn import functional as F

from models import inflate
from .resnets1 import resnet50_s1
from .clone_2d import CloneBottleneck
from .TSE import TSE
from .TSB import TSB



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        # init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class Bottleneck3d(nn.Module):

    def __init__(self, bottleneck2d, inflate_time=False):
        super(Bottleneck3d, self).__init__()

        if inflate_time == True:
            self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=3, time_padding=1, center=True)
        else:
            self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)
        self.conv2 = inflate.inflate_conv(bottleneck2d.conv2, time_dim=1)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = self._inflate_downsample(bottleneck2d.downsample)
        else:
            self.downsample = None

    def _inflate_downsample(self, downsample2d, time_stride=1):
        downsample3d = nn.Sequential(
            inflate.inflate_conv(downsample2d[0], time_dim=1, 
                                 time_stride=time_stride),
            inflate.inflate_batch_norm(downsample2d[1]))
        return downsample3d

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TCLNet(nn.Module):

    def __init__(self, num_classes, use_gpu, loss={'xent'}):
        super(TCLNet, self).__init__()
        self.loss = loss
        self.use_gpu = use_gpu
        resnet2d = resnet50_s1(pretrained=True)

        self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)
        
        self.layer1 = self._inflate_reslayer(resnet2d.layer1)
        self.layer2 = self._inflate_reslayer(resnet2d.layer2, enhance_idx=[3], channels=512)
        self.layer3 = self._inflate_reslayer(resnet2d.layer3)
        layer4 = nn.Sequential(resnet2d.layer4[0], resnet2d.layer4[1])

        branch = nn.ModuleList([CloneBottleneck(resnet2d.layer4[-1]) for _ in range(2)])

        self.TSE_Module = TSE(layer4=layer4, branch=branch, use_gpu=use_gpu)

        bn = []
        for _ in range(2):
            add_block = nn.BatchNorm1d(2048)
            add_block.apply(weights_init_kaiming)
            bn.append(add_block)
        self.bn = nn.ModuleList(bn)

        classifier = []
        for _ in range(2):
            add_block = nn.Linear(2048, num_classes)
            add_block.apply(weights_init_classifier)
            classifier.append(add_block)
        self.classifier = nn.ModuleList(classifier)
        
    def _inflate_reslayer(self, reslayer2d, enhance_idx=[], channels=0):
        reslayers3d = []
        for i, layer2d in enumerate(reslayer2d):
            layer3d = Bottleneck3d(layer2d)
            reslayers3d.append(layer3d)

            if i in enhance_idx:
                TSB_Module = TSB(in_channels=channels, use_gpu=self.use_gpu)
                reslayers3d.append(TSB_Module)

        return nn.Sequential(*reslayers3d)

    def pool(self, x):
        x = F.max_pool2d(x, x.size()[2:]) 
        x = x.view(x.size(0), -1) #[b, c]
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        b, c, t, h, w = x.size()

        x1_list = self.TSE_Module(x[:, :, :2]) 
        x2_list = self.TSE_Module(x[:, :, 2:])

        x1 = self.pool(x1_list[0])
        x2 = self.pool(x1_list[1])
        x3 = self.pool(x2_list[0])
        x4 = self.pool(x2_list[1])

        x1 = torch.stack((x1, x3), 1) #[b, 2, c]
        x2 = torch.stack((x2, x4), 1)

        x1 = x1.mean(1) #[b, c]
        x2 = x2.mean(1)

        if not self.training:
            x = torch.cat((x1, x2), 1) #[b, c * 2]
            return x

        f1 = self.bn[0](x1)
        f2 = self.bn[1](x2)

        y1 = self.classifier[0](f1)
        y2 = self.classifier[1](f2)

        if self.loss == {'xent'}:
            return [y1, y2]
        elif self.loss == {'xent', 'htri'}:
            return [y1, y2], [f1, f2]
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
