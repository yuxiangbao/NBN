"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ResNet import BasicBlock, Bottleneck, conv3x3

import pdb


class ResNet(nn.Module):

    def __init__(self, block, layers, num_experts, dropout=None, num_classes=1000, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, share_layer3=False, returns_feat=False, s=30):
        self.inplanes = 64
        self.num_experts = num_experts
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.inplanes = self.next_inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.inplanes = self.next_inplanes

        self.share_layer3 = share_layer3

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 192
            else:
                layer3_output_dim = 256

        if layer4_output_dim is None:
            if reduce_dimension:
                layer4_output_dim = 384
            else:
                layer4_output_dim = 512

        if self.share_layer3:
            self.layer3 = self._make_layer(block, layer3_output_dim, layers[2], stride=2)
        else:
            self.layer3s = nn.ModuleList([self._make_layer(block, layer3_output_dim, layers[2], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.layer4s = nn.ModuleList([self._make_layer(block, layer4_output_dim, layers[3], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        self.use_dropout = True if dropout else False

        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        
        self.linears = nn.ModuleList([nn.Linear(layer4_output_dim * block.expansion, num_classes) for _ in range(num_experts)])
        s = 1
        
        self.s = s

        self.returns_feat = returns_feat

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()
                    count += 1

        if count > 0:
            print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.next_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.next_inplanes, planes))

        return nn.Sequential(*layers)

    def _separate_part(self, x, ind):
        if not self.share_layer3:
            x = (self.layer3s[ind])(x)
        x = (self.layer4s[ind])(x)

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)

        if self.use_dropout:
            x = self.dropout(x)

        self.feat.append(x)
        x = (self.linears[ind])(x)
        x = x * self.s
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        if self.share_layer3:
            x = self.layer3(x)

        outs = []
        self.feat = []
        for ind in range(self.num_experts):
            outs.append(self._separate_part(x, ind))

        final_out = torch.stack(outs, dim=1).mean(dim=1)

        if self.returns_feat:
            return {
                "output": final_out, 
                "feat": torch.stack(self.feat, dim=1),
                "logits": torch.stack(outs, dim=1)
            }
        else:
            return final_out

def resnet50_ride(num_classes=1000,
                 reduce_dimension=False,
                 returns_feat=True,
                 num_experts=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], 
                 num_classes=num_classes, 
                 reduce_dimension=reduce_dimension, 
                 num_experts=num_experts,
                 returns_feat=returns_feat)