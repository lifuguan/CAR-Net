import torch.nn as nn
import torch
from functools import partial
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary

class convBlock(nn.Module):
    '''
    activation: 默认使用激活函数
    in_channels: 输入的feature map的通道数
    filters: 输出的out_channels的通道数
    kernel_size: 卷积核的大小
    stride: 步长
    padding: 补0（zero-padding）的数量
    '''
    def __init__(self, in_channels, filters, kernel_size, stride = 1, activation = True):
        '''
        super()函数：用来解决类的继承问题
        此处：super会将convBlock的对象转化成nn.Module的对象，从而调用nn.Module的__init__()初始化函数
        '''
        super(convBlock, self).__init__() # run : nn.Module.__init__()
        self.activation = activation
        # // : 代表整除法
        self.conv = nn.Conv2d(in_channels, filters, kernel_size, stride = stride, padding = kernel_size//2)
        self.norm = nn.BatchNorm2d(filters)

    # 前向传播函数
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            return F.relu(x)
        else:
            return x

class residualBlock(nn.Module):
    def __init__(self, in_channels, filters, size = 3):
        super(residualBlock, self).__init__()

        self.norm = nn.BatchNorm2d(in_channels)
        self.conv1 = convBlock(in_channels, filters, size)
        self.conv2 = convBlock(filters, filters, size, activation=False)
    
    # 前向传播函数
    def forward(self, x):
        residual = x  
        # 1. ReLU激活函數
        x = F.relu(x)
        # 2. Batch Normalization 批量归一化函数
        x = self.norm(x)
        # 3. 卷积层
        x = self.conv1(x)
        # 4. 卷积层
        x = self.conv2(x)
        #x += residual
        return x 

class deconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 2, stride = 2):
        super(deconvBlock, self).__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride)

    def forward(self, x1, x2):
        # x2为skip-connection
        xd = self.deconv(x1)
        x = torch.cat([xd, x2], dim = 1)
        return x

class UnetResnet(nn.Module):

    def __init__(self, filters = 16, dropout = 0.5):
        super(UnetResnet, self).__init__()
        
        '''左边第一层
        1. 卷积层
        2. 残差层
        3. 残差层
        4. ReLU激活函数
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, filters, 3, padding = 1),
            residualBlock(filters, filters),
            residualBlock(filters, filters),
            nn.ReLU()
        )
        
        '''左边第二层
        1. 最大池化层（downsampling）
        2. Dropout层
        3. 残差层
        4. 残差层
        5. ReLU激活函数
        '''
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, 2), # 设置参数为(2, 2)，即两倍降采样
            nn.Dropout(dropout/2), # 前向传播时随机干掉 50%/2 = 25%的神经元
            # in_channel: 16
            # out_channel: 32
            # kernel_size: 3*3
            # padding: 1
            nn.Conv2d(filters, filters * 2, 3, padding = 1), 
            residualBlock(filters * 2, filters * 2),
            residualBlock(filters * 2, filters * 2),
            nn.ReLU()
        )
        '''左边第三层
        '''
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),
            # in_channel: 32
            # out_channel: 64
            # kernel_size: 3*3
            # padding: 1
            nn.Conv2d(filters * 2, filters * 4, 3, padding = 1),
            residualBlock(filters * 4, filters * 4),
            residualBlock(filters * 4, filters * 4),
            nn.ReLU()
        )
        '''左边第四层
        '''
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),
            # in_channel: 64
            # out_channel: 128
            # kernel_size: 3*3
            # padding: 1
            nn.Conv2d(filters * 4, filters * 8, 3, padding = 1),
            residualBlock(filters * 8, filters * 8),
            residualBlock(filters * 8, filters * 8),
            nn.ReLU()
        )
            
        '''中间层
        1. 最大池化层（downsampling）
        2. Dropout层
        3. 残差层
        4. 残差层
        5. ReLU激活函数
        '''
        self.middle = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),
            # in_channel: 128
            # out_channel: 256
            # kernel_size: 3*3
            # padding: 1
            nn.Conv2d(filters * 8, filters * 16, 3, padding = 3//2),
            residualBlock(filters * 16, filters * 16),
            residualBlock(filters * 16, filters * 16),
            nn.ReLU()
        )
        
        self.deconv4 = deconvBlock(filters * 16, filters * 8, 2)
        self.upconv4 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(filters * 16, filters * 8, 3, padding = 1),
            residualBlock(filters * 8, filters * 8),
            residualBlock(filters * 8, filters * 8),
            nn.ReLU()
        )
  

        self.deconv3 = deconvBlock(filters * 8, filters * 4, 3)
        self.upconv3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(filters * 8, filters * 4, 3, padding = 1),
            residualBlock(filters * 4, filters * 4),
            residualBlock(filters * 4, filters * 4),
            nn.ReLU()
        )
        
        self.deconv2 = deconvBlock(filters * 4, filters * 2, 2)
        self.upconv2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(filters * 4, filters * 2, 3, padding = 1),
            residualBlock(filters * 2, filters * 2),
            residualBlock(filters * 2, filters * 2),
            nn.ReLU()
        )

        self.deconv1 = deconvBlock(filters * 2, filters, 3)
        self.upconv1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(filters * 2, filters, 3, padding = 1),
            residualBlock(filters, filters),
            residualBlock(filters, filters),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Conv2d(filters, 1, 3, padding = 1)
        )

    def forward(self, x):
        conv1 = self.conv1(x) 
        # 101 -> 50
        conv2 = self.conv2(conv1) 
        # 50 -> 25
        conv3 = self.conv3(conv2) 
        # 25 -> 12
        conv4 = self.conv4(conv3) 
        # 12 - 6
        x = self.middle(conv4) 
        
        # 6 -> 12
        x = self.deconv4(x, conv4) # 逆卷积middle + 跳转连接conv4
        x = self.upconv4(x)
        # 12 -> 25
        x = self.deconv3(x, conv3) # 逆卷积upconv4 + 跳转连接conv3
        x = self.upconv3(x)
        # 25 -> 50
        x = self.deconv2(x, conv2) # 逆卷积upconv3 + 跳转连接conv2
        x = self.upconv2(x)
        # 50 -> 101
        x = self.deconv1(x, conv1) # 逆卷积upconv2 + 跳转连接conv1
        x = self.upconv1(x)  # 逆卷积upconv1

        return x
