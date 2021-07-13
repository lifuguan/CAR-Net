'''
Author: your name
Date: 2021-04-03 22:36:24
LastEditTime: 2021-06-12 13:40:03
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /leo/unetzoo/design/attention_design_two.py
'''
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from functools import partial
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
import os

__all__ = ["design_access"]

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        return self.conv(input)

class AttentionAccess(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AttentionAccess, self).__init__()
        resnet = models.resnet34(pretrained=True)
        
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.pool5 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)


        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.attention1 = AttentionBlock(64, 64)
        self.attention2 = AttentionBlock(128, 128)
        self.attention3 = AttentionBlock(256, 256)
        self.attention4 = AttentionBlock(512, 512)

        self.finaldeconv1 = nn.ConvTranspose2d(960, 32, 4, 2, 1)
        # self.finaldeconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.se_block = SELayer(channel=32)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)
    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        # x = self.firstmaxpool(x)
        en1 = self.encoder1(x)    # 64  , 512 , 512
        en2 = self.encoder2(en1)  # 128 , 256 , 256
        en3 = self.encoder3(en2)  # 256 , 128 , 128
        en4 = self.encoder4(en3)  # 512 , 64  , 64

        pool5 = self.pool5(en4)
        en5 = self.conv5(pool5)   #1024 , 16  , 16

        up_6 = self.up6(en5)
        merge6 = self.attention4(en4, en5)
        c6 = self.conv6(torch.cat([up_6, merge6], dim = 1))
        up_7 = self.up7(c6)
        merge7 = self.attention3(en3, c6)
        c7 = self.conv7(torch.cat([up_7, merge7], dim = 1))
        up_8 = self.up8(c7)
        merge8 = self.attention2(en2, c7)
        c8 = self.conv8(torch.cat([up_8, merge8], dim = 1))
        up_9 = self.up9(c8)
        merge9 = self.attention1(en1, c8)
        c9 = self.conv9(torch.cat([up_9, merge9], dim = 1))

        c6_up =  F.interpolate(c6, scale_factor=8, mode='bilinear', align_corners=True)
        c7_up =  F.interpolate(c7, scale_factor=4, mode='bilinear', align_corners=True)
        c8_up =  F.interpolate(c8, scale_factor=2, mode='bilinear', align_corners=True)
        up = torch.cat([c6_up, c7_up, c8_up, c9], dim=1)
        out = self.finaldeconv1(up)
        out = self.se_block(out)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)

     
nonlinearity = partial(F.relu, inplace=True)
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, type = 1):
        super(AttentionBlock, self).__init__()
        self.b_up_2 = nn.ConvTranspose2d(in_channels * 2, in_channels, 2, stride=2)

        self.att_type = type
        self.ChannelGate = ChannelGate(in_channels, reduction_ratio=16)
        self.SpatialGate = SpatialGate()

        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
    def forward(self, x, b):
        b = self.b_up_2(b)
        b1 = self.conv3(b)
        channel1 = self.ChannelGate(x) + self.ChannelGate(b)
        channel1 = self.SpatialGate(channel1)

        if self.att_type == 1:
            channel2 = self.conv2(x) + b1
            channel2 = self.SpatialGate(channel2)
            channel3 = self.SpatialGate(b1)
            psi = channel1 + channel2 + channel3

        if self.att_type == 2:
            channel3 = self.SpatialGate(b1)
            psi = channel1 + channel3

        if self.att_type == 3:
            psi = channel1

        return x * psi

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 16, pool_types = ['avg', 'max']):
        '''
        @description: 通道注意力
        @param in_channel: 输入通道数； reduction_ratio: 神经元缩放大小； pool_types: 池化类型
        '''
        super(ChannelGate, self).__init__()
        self.in_channels = in_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        
        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale      

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale    

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

"""print layers and params of network"""
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = AttentionBlock(64, 64).to(device)
    model = AttentionAccess(3, 1).to(device)
    summary(model,(3, 512, 512))
