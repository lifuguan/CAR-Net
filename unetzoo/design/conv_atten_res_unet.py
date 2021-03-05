# -*- coding: utf-8 -*-

#
# Title: Conv_Atten_Res_UNet
# Author: Cheng Yong
# Notes: 添加了注意力(AttentionConv)之后并直接和输入相加
# Date: 2021-01-30
#

import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from common.attention.attention import AttentionConv


__all__ = ["Conv_Atten_Res_UNet"]


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.out_ch = out_ch
        self.aconv = nn.Conv2d(in_ch, self.out_ch, 3, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            AttentionConv(out_ch, out_ch, kernel_size=3, padding=1),  # 添加AttentionConv
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        shortcut = self.aconv(input)
        return F.relu(torch.add(self.conv(input), shortcut))


class Conv_Atten_Res_UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_Atten_Res_UNet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        # print(x.shape)
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        # print(p1.shape)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        # print(p2.shape)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        # print(p3.shape)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        # print(p4.shape)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out


"""print network information and params"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv_Atten_Res_UNet(3, 1).to(device)
    print(model)
    summary(model, (3, 224, 224))  # Succeed!
