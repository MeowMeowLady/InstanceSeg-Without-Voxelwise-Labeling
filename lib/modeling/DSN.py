# -*- coding: utf-8 -*-
"""
Created on 18-11-16 下午3:34
IDE PyCharm 

@author: Meng Dong
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from core.config import cfg


class dsn_body(nn.Module):
    def __init__(self):
        super(dsn_body, self).__init__()

        self.conv1a = nn.Conv3d(1, 32, 5, 1, 2, bias = True)
        self.bn1a=nn.BatchNorm3d(32, momentum = 0.001, affine=True)
        self.pool1 = nn.MaxPool3d(2, 2, padding = 0)
        self.conv2a = nn.Conv3d(32, 64, 3, 1, 1, bias = True)
        self.bn2a = nn.BatchNorm3d(64, momentum = 0.001, affine = True)
        self.conv2b = nn.Conv3d(64, 64, 3, 1, 1, bias = True)
        self.bn2b = nn.BatchNorm3d(64, momentum = 0.001, affine = True)
        self.pool2 = nn.MaxPool3d(2, 2, padding = 0)
        self.conv3a = nn.Conv3d(64, 128, 3, 1, 1, bias = True)
        self.bn3a = nn.BatchNorm3d(128, momentum = 0.001, affine = True)
        self.conv3b = nn.Conv3d(128, 128, 3, 1, 1, bias = True)
        self.bn3b = nn.BatchNorm3d(128, momentum = 0.001, affine = True)
        if cfg.RPN.STRIDE == 8:
            self.pool3 = nn.MaxPool3d(2, 2, padding = 0)
            self.conv4a = nn.Conv3d(128, 256, 3, 1, 1, bias = True)
            self.bn4a = nn.BatchNorm3d(256, momentum = 0.001, affine = True)
            self.conv4b = nn.Conv3d(256, 256, 3, 1, 1, bias = True)
            self.bn4b = nn.BatchNorm3d(256, momentum = 0.001, affine = True)
            self.dim_out = 256
        else:
            self.dim_out = 128

        self.__weight_init()
        self.spatial_scale = 1./cfg.RPN.STRIDE
    #weight initialization
    def __weight_init(self):
        for m in self.modules():
            m.name = m.__class__.__name__
            if m.name.find('Conv')!=-1:
                nn.init.normal_(m.weight, std = 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            if m.name.find('BatchNorm3d')!=-1:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)



    def forward(self, main):
        main = self.pool1(nnf.relu(self.bn1a(self.conv1a(main))))
        main = nnf.relu(self.bn2a(self.conv2a(main)))
        main = nnf.relu(self.bn2b(self.conv2b(main)))
        main = self.pool2(main)
        main = nnf.relu(self.bn3a(self.conv3a(main)))
        main = nnf.relu(self.bn3b(self.conv3b(main)))
        if cfg.RPN.STRIDE == 8:
            main = self.pool3(main)
            main = nnf.relu(self.bn4a(self.conv4a(main)))
            main = nnf.relu(self.bn4b(self.conv4b(main)))
        return main
