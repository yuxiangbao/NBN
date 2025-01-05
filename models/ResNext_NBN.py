"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.LRClassifier import (Linear_fcbn, Linear_fcbn_notaffine)
import pdb

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, dim):
        super().__init__(dim)
        
    def scale(self, alpha):
        self.w = self.weight / (torch.norm(self.weight)+1e-6) * alpha
        if self.bias is not None:
            self.b = self.bias / (torch.norm(self.bias)+1e-6) * alpha
        else:
            self.b = self.bias
            
    def forward(self, input):
        self._check_input_dim(input)

        w = self.w if hasattr(self, 'w') else self.weight
        b = self.b if hasattr(self, 'b') else self.bias
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            w,
            b,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, is_last=False):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               groups=groups, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.is_last = is_last

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

class ResNext_scale(nn.Module):

    def __init__(self, block, layers, fc_type="Linear", groups=1, width_per_group=64,
                 dropout=None,
                 use_glore=False, use_gem=False, num_classes=1000, num_tasks=3):
        self.inplanes = 64
        super().__init__()

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        self.use_dropout = True if dropout else False

        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        self.fc_type = fc_type

        if fc_type=="Linear_fcbn":
            self.fc = Linear_fcbn(512*block.expansion, num_classes)
        elif fc_type=="Linear_fcbn_notaffine":
            self.fc = Linear_fcbn_notaffine(512*block.expansion, num_classes)
        elif fc_type=="Linear":
            self.fc= nn.Linear(512*block.expansion, num_classes)
        
        self.init_scale_factor()
    
    def init_scale_factor(self):
        module_list = []
        for n, m in self.named_modules():
            if hasattr(m, "scale"):
                module_list.append(n)
        self.scale_factor = nn.Parameter(torch.ones((len(module_list),)))
        self.module_list = module_list

    def scale_modules(self):
        pass

    def _make_layer(self, block, planes, blocks, stride=1, is_last=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            groups=self.groups, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups, base_width=self.base_width,
                                is_last=(is_last and i == blocks-1)))

        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        self.scale_modules()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
    
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc(x, **kwargs)
       
        return x


class ResNeXt_NBN(ResNext_scale):
    def scale_modules(self):
        for n, m in self.named_modules():
            if hasattr(m, "scale") and isinstance(m, BatchNorm2d):
                if 'layer4' in n:
                    m.scale(self.scale_factor[0])


def resnext50_nbn(num_classes=1000, **kw):
    print('Loading Scratch ResNext 50  Model.')
    resnext = ResNeXt_NBN(Bottleneck, [3, 4, 6, 3], fc_type="Linear", 
                    dropout=None, groups=32, width_per_group=4, 
                    use_glore=False, use_gem=False, num_classes=num_classes)
    return resnext