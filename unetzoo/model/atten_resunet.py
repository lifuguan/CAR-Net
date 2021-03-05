# -*- coding: utf-8 -*-

#
# Title: UNet and cosegmentation for siamese detector
# Author: xiaoyu
# Source: https://github.com/xiaoyuvision/Attention-Unet-For-Co-segmentation/blob/master/module.py
# Date: 2021-02-06
#


import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torchvision import models

class Dblock_more_dilate(nn.Module):
    
    def __init__(self,channel):
        super(Dblock_more_dilate, self).__init__()
        #self.nonlinearity =nn.ReLU(inplace=True)
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
       
    def forward(self, x):
        dilate1_out = F.relu(self.dilate1(x),inplace=True)
        dilate2_out = F.relu(self.dilate2(dilate1_out),inplace=True)
        dilate3_out = F.relu(self.dilate3(dilate2_out),inplace=True)
        dilate4_out = F.relu(self.dilate4(dilate3_out),inplace=True)
        dilate5_out = F.relu(self.dilate5(dilate4_out),inplace=True)
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out

class Dblock(nn.Module):
    
    def __init__(self,channel):
        
        super(Dblock, self).__init__()
        #self.nonlinearity = F.relu(inplace=True)
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
       
    def forward(self, x):
        dilate1_out = F.relu(self.dilate1(x),inplace=True)
        dilate2_out = F.relu(self.dilate2(dilate1_out),inplace=True)
        dilate3_out = F.relu(self.dilate3(dilate2_out),inplace=True)
        dilate4_out = F.relu(self.dilate4(dilate3_out),inplace=True)
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out# + dilate5_out
        return out

class DecoderBlock(nn.Module):
    
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()
        

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        #self.relu1 = self.nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        #self.relu2 = self.nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        #self.relu3 = self.nonlinearity
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        
        
        #x = self.relu1(x)
        x=F.relu(x,inplace=True)
        
        x = self.deconv2(x)
        x = self.norm2(x)
        
        #x = self.relu2(x)
        x=F.relu(x,inplace=True)
        
        x = self.conv3(x)
        x = self.norm3(x)
        
        #x = self.relu3(x)
        x=F.relu(x,inplace=True)
        
        
        return x
    
class Attention(nn.Module):
    
    def __init__(self,feature,tunnel,channel):
        super(Attention,self).__init__()
        self.feature=feature
        self.tunnel=tunnel
        self.channel=channel
        
        
        self.linear1=nn.Linear(self.feature,self.tunnel)
        self.linear2=nn.Linear(self.tunnel,self.feature)
        self.upsample=nn.Upsample(self.channel)
        self.avg_pool=nn.AdaptiveAvgPool2d((1, 1))
        
        
    def forward(self,x):
        x=self.avg_pool(x)
        x=F.tanh(x.view(-1,self.feature))
        x=self.linear1(x)
        x=F.tanh(x)
        x=self.linear2(x)
        x=F.sigmoid(x)
        x=x.view(-1,self.feature,1,1)
        x=self.upsample(x)
        return(x)

class Atten_ResUNet(nn.Module):
    
    def __init__(self,num_classes=1, num_channels=3):
        super(Atten_ResUNet, self).__init__()

        filters = [64, 128, 256, 512]
        #resnet = models.resnet34(pretrained=True)
        resnet = models.resnet18(pretrained=False)
        resnet.load_state_dict(torch.load('E:/model/pytorch/resnet18-5c106cde.pth'))
        
        #resnet.load_state_dict(torch.load('weights/resnet34-333f7ec4.pth'))
        #example=torch.ones((1,3,224,224))
        #resnet=torch.jit.trace(resnet,example)
        #resnet.load_state_dict(torch.load('weights/resnet34-333f7ec4.pth'))
        
        self.attention1=Attention(64,128,192)
        self.attention2=Attention(128,256,96)
        self.attention3=Attention(256,512,48)
        self.attention4=Attention(512,1024,24)
        
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        #self.finalrelu1 = self.nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        #self.finalrelu2 = self.nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
    
    def forward(self, x,y):
        # Encoder
        x = self.firstconv(x)#(2,64,383,383)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)#(2,64,192,192)

       
        e1 = self.encoder1(x)#(2,64,192,192)
        #print('.........',e1.shape)
        attention_e1=self.attention1(e1)
        e2 = self.encoder2(e1)#(2,128,96,96)
        #print('........',e2.shape)
        attention_e2=self.attention2(e2)
        e3 = self.encoder3(e2)#(2,256,48,48)
        attention_e3=self.attention3(e3)
        e4 = self.encoder4(e3)#(2,512,24,24)
        attention_e4=self.attention4(e4)
        
        
        y = self.firstconv(y)#(2,64,383,383)
        y = self.firstbn(y)
        y = self.firstrelu(y)
        y = self.firstmaxpool(y)
        
        f1 = self.encoder1(y)#(2,256,192,192)
        attention_f1=self.attention1(f1)
        f2 = self.encoder2(e1)#(2,512,96,96)
        attention_f2=self.attention2(f2)
        f3 = self.encoder3(e2)#(2,1024,48,48)
        attention_f3=self.attention3(f3)
        f4 = self.encoder4(e3)#(2,2048,24,24)
        attention_f4=self.attention4(f4)
        
        # Center
        e4 = self.dblock(attention_f4*e4)
        f4 = self.dblock(attention_e4*f4)

        # Decoder
        d4 = self.decoder4(e4) + attention_f3*e3
        d3 = self.decoder3(d4) + attention_f2*e2
        d2 = self.decoder2(d3) + attention_f1*e1
        d1 = self.decoder1(d2)
        
        g4 = self.decoder4(f4) + attention_e3*f3
        g3 = self.decoder3(g4) + attention_e2*f2
        g2 = self.decoder2(g3) + attention_e1*f1
        g1 = self.decoder1(g2)
        
        out_x = self.finaldeconv1(d1)
        #out = self.finalrelu1(out)
        out_x = F.relu(out_x,inplace=True)
        
        out_x = self.finalconv2(out_x)
        #out = self.finalrelu2(out)
        out_x = F.relu(out_x,inplace=True)
        out_x = self.finalconv3(out_x)
        
        out_y = self.finaldeconv1(g1)
        #out = self.finalrelu1(out)
        out_y = F.relu(out_y,inplace=True)
        
        out_y = self.finalconv2(out_y)
        #out = self.finalrelu2(out)
        out_y = F.relu(out_y,inplace=True)
        out_y = self.finalconv3(out_y)

        return F.sigmoid(out_x),F.sigmoid(out_y)


"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Atten_ResUNet().to(device)
    print(model)
    # summary(model,(1, 576, 576))    # Fails

