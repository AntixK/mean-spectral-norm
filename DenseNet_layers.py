from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from SNlayers import SNConv2d, SNLinear, MeanSpectralNorm

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features)),
        self.add_module("relu1", nn.ReLU(inplace=True)),
        self.add_module(
            "conv1", nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        ),
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module("relu2", nn.ReLU(inplace=True)),
        self.add_module(
            "conv2", nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        ),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _SNDenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_SNDenseLayer, self).__init__()
        self.add_module("relu1", nn.ReLU(inplace=True)),
        self.add_module(
            "conv1", SNConv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        ),
        self.add_module("relu2", nn.ReLU(inplace=True)),
        self.add_module(
            "conv2", SNConv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        ),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_SNDenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _MSNDenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_MSNDenseLayer, self).__init__()
        self.add_module("norm1", MeanSpectralNorm(num_input_features)),
        self.add_module("relu1", nn.ReLU(inplace=True)),
        self.add_module(
            "conv1", SNConv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        ),
        self.add_module("norm2", MeanSpectralNorm(bn_size * growth_rate)),
        self.add_module("relu2", nn.ReLU(inplace=True)),
        self.add_module(
            "conv2", SNConv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        ),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_MSNDenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))

class _SNTransition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_SNTransition, self).__init__()
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", SNConv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))

class _MSNTransition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_MSNTransition, self).__init__()
        self.add_module("norm", MeanSpectralNorm(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", SNConv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)
