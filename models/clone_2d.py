from __future__ import absolute_import

import torch
import torch.nn as nn

class CloneBottleneck(nn.Module):
    def __init__(self, bottleneck2d):
        super(CloneBottleneck, self).__init__()
        self.conv1 = clone_conv(bottleneck2d.conv1)
        self.bn1 = clone_batch_norm(bottleneck2d.bn1)
        self.conv2 = clone_conv(bottleneck2d.conv2)
        self.bn2 = clone_batch_norm(bottleneck2d.bn2)
        self.conv3 = clone_conv(bottleneck2d.conv3)
        self.bn3 = clone_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = self._clone_downsample(bottleneck2d.downsample)
        else:
            self.downsample = None

    def _clone_downsample(self, downsample2d):
        downsample2d_clone = nn.Sequential(
                clone_conv(downsample2d[0]),
                clone_batch_norm(downsample2d[1]))
        return downsample2d_clone

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


def clone_conv(conv2d):
    # To preserve activations, padding should be by continuity and not zero
    # or no padding in time dimension
    conv2d_clone = nn.Conv2d(
        conv2d.in_channels,
        conv2d.out_channels,
        conv2d.kernel_size,
        padding=conv2d.padding,
        dilation=conv2d.dilation,
        stride=conv2d.stride)

    # Assign new params
    weight_2d = conv2d.weight.data.clone()
    conv2d_clone.weight = nn.Parameter(weight_2d)
    assert (conv2d.bias is None)
    conv2d_clone.bias = conv2d.bias
    return conv2d_clone

def clone_batch_norm(batch2d):
    # In pytorch 0.2.0 the 2d and 3d versions of batch norm
    # work identically except for the check that verifies the
    # input dimensions

    batch2d_clone = nn.BatchNorm2d(batch2d.num_features)
    weight_2d = batch2d.weight.data.clone()
    bias_2d = batch2d.bias.data.clone()
    batch2d_clone.weight = nn.Parameter(weight_2d)
    batch2d_clone.bias = nn.Parameter(bias_2d)

    # retrieve 3d _check_input_dim function
    batch2d_clone._check_input_dim = batch2d._check_input_dim
    return batch2d_clone


if __name__ == '__main__':
    a = nn.Conv2d(2, 2, kernel_size=1, bias=False) 
    b = clone_conv(a)
    a.weight[1, 1] = 0
    print(a.weight)
    print(a.bias)
    print(b.weight)
    print(b.bias)
