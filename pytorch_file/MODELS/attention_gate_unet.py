import torch 
import torch.nn as nn
import torch.nn.functional as F
'''
Essential part of UNet model
'''

class conv_block(nn.Module):
    '''
    activation: 默认使用激活函数
    in_channels: 输入的feature map的通道数
    out_channels: 输出的out_channels的通道数
    kernel_size: 卷积核的大小
    stride: 步长
    padding: 补0（zero-padding）的数量
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=True):
        '''
        super()函数：用来解决类的继承问题
        此处：super会将conv_block的对象转化成nn.Module的对象，从而调用nn.Module的__init__()初始化函数
        '''
        super(conv_block, self).__init__()  # run : nn.Module.__init__()
        self.activation = activation
        # // : 代表整除法
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=kernel_size//2)
        self.norm = nn.BatchNorm2d(out_channels)

    # 前向传播函数
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            return F.relu(x)
        else:
            return x


class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            # 升采样，使用‘bilinear’模式（upsample和convTransposed的区别是upsample不需要训练参数）
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x



class Attention_Block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_Block, self).__init__()
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.sigma_1 = nn.ReLU(inplace=True)

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1)
        )
        self.sigma_2 = nn.Sigmoid()
        self.test_print = True

    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        sigma1 = self.sigma_1(g1+x1)
        psi = self.psi(sigma1)

        sigma2 = self.sigma_2(psi)

        res = x*sigma2
        return res

'''
UNet Model 
'''


class AttenGateUNet(nn.Module):
    def __init__(self, img_channels=1, num_classes=1):
        super(AttenGateUNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_channels=img_channels, out_channels=64)
        self.Conv2 = conv_block(in_channels=64, out_channels=128)
        self.Conv3 = conv_block(in_channels=128, out_channels=256)
        self.Conv4 = conv_block(in_channels=256, out_channels=512)
        self.Conv5 = conv_block(in_channels=512, out_channels=1024)

        self.Up5 = up_conv(in_channels=1024, out_channels=512)
        self.Att5 = Attention_Block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(in_channels=1024, out_channels=512)

        self.Up4 = up_conv(in_channels=512, out_channels=256)
        self.Att4 = Attention_Block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(in_channels=512, out_channels=256)

        self.Up3 = up_conv(in_channels=256, out_channels=128)
        self.Att3 = Attention_Block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(in_channels=256, out_channels=128)

        self.Up2 = up_conv(in_channels=128, out_channels=64)
        self.Att2 = Attention_Block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(in_channels=128, out_channels=64)

        self.Conv_1x1 = nn.Conv2d(
            64, num_classes, kernel_size=1, stride=1, padding=0)

        self.test_print = True

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        self.test_print = False
        return d1
