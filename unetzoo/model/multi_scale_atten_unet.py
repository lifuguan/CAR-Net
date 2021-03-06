# -*- coding: utf-8 -*-

#
# Title: NO
# Arxiv: NO
# Author: liuxiangqiang
# Source: https://github.com/Edward-LXQ/Multi-Scale-Fuse-With-Attention-Model/blob/main/model
# Date: 2021-02-09
#

from __future__ import print_function, division
import torch.nn as nn
import torch.utils.data
import torch
from torchsummary import summary


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block_3(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block_3, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_5(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block_5, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_7(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block_7, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class fuse_block(nn.Module):
    def __init__(self, F_conv3, F_conv5, F_conv7, out_ch):
        super(fuse_block, self).__init__()

        self.conv3 = nn.Sequential(
            nn.Conv2d(F_conv3, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(F_conv5, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(F_conv7, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(out_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, t_conv3, t_conv5, t_conv7):
        conv_3 = self.conv3(t_conv3)
        conv_5 = self.conv5(t_conv5)
        conv_7 = self.conv7(t_conv7)
        psi = self.relu(conv_3 + conv_5 + conv_7)
        psi = self.psi(psi)
        out = t_conv3 * psi
        return out


class mutil_conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, F_int):
        super(mutil_conv_block, self).__init__()

        self.conv3 = conv_block_3(in_ch, out_ch)
        self.conv5 = conv_block_5(in_ch, out_ch)
        self.conv7 = conv_block_7(in_ch, out_ch)
        self.fuse = fuse_block(in_ch, in_ch, in_ch, out_ch)

    def forward(self, x):
        conv_3 = self.conv3(x)
        conv_5 = self.conv5(x)
        conv_7 = self.conv7(x)
        out = self.fuse(conv_3, conv_5, conv_7)
        return out


class Multi_Scale_Atten_UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(Multi_Scale_Atten_UNet, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = mutil_conv_block(in_ch, filters[0], 16)
        self.Conv2 = mutil_conv_block(filters[0], filters[1], filters[0])
        self.Conv3 = mutil_conv_block(filters[1], filters[2], filters[1])
        self.Conv4 = mutil_conv_block(filters[2], filters[3], filters[2])
        self.Conv5 = mutil_conv_block(filters[3], filters[4], filters[3])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block_3(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block_3(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block_3(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block_3(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        d1 = self.active(out)
        return d1


"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Multi_Scale_Atten_UNet(in_ch=1, out_ch=1).to(device)
    print(model)
    # summary(model,(1,128,128))     # Fails


