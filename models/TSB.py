from __future__ import absolute_import
from __future__ import division

import torch
import math
import numpy as np
from torch import nn
from torch.nn import functional as F


class TSB(nn.Module):
    def __init__(self, in_channels, use_gpu=False, **kwargs):
        super(TSB, self).__init__()
        self.in_channels = in_channels
        self.use_gpu = use_gpu
        self.patch_size = 2
        
        self.W = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels,
                    kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.in_channels)
        )

        self.pool = nn.AvgPool3d(kernel_size=(1, self.patch_size, self.patch_size), 
                stride=(1, 1, 1), padding=(0, self.patch_size//2, self.patch_size//2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        nn.init.constant_(self.W[1].weight.data, 0.0)
        nn.init.constant_(self.W[1].bias.data, 0.0)


    def forward(self, x):
        b, c, t, h, w = x.size()
        inputs = x

        query = x.view(b, c, t, -1).mean(-1) 
        query = query.permute(0, 2, 1) 

        memory = self.pool(x) 
        if self.patch_size % 2 == 0:
            memory = memory[:, :, :, :-1, :-1]

        memory = memory.contiguous().view(b, 1, c, t * h * w) 

        query = F.normalize(query, p=2, dim=2, eps=1e-12)
        memory = F.normalize(memory, p=2, dim=2, eps=1e-12)
        f = torch.matmul(query.unsqueeze(2), memory) * 5
        f = f.view(b, t, t, h * w) 

        # mask the self-enhance
        mask = torch.eye(t).type(x.dtype) 
        if self.use_gpu: mask = mask.cuda()
        mask = mask.view(1, t, t, 1)

        f = (f - mask * 1e8).view(b, t, t * h * w)
        f = F.softmax(f, dim=-1)

        y = x.view(b, c, t * h * w)
        y = torch.matmul(f, y.permute(0, 2, 1)) 
        y = self.W(y.view(b * t, c, 1, 1))
        y = y.view(b, t, c, 1, 1)
        y = y.permute(0, 2, 1, 3, 4)
        z = y + inputs

        return z
