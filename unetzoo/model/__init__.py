# -*- coding: utf-8 -*-

#
# Title: generate model instance.
# Source:
# Date: 2021-02-26
#


from model.segnet import SegNet
from model.fcn import get_fcn8s, get_fcn32s
from model.cenet import CE_Net_
from model.unet import UNet
from model.attention_gate_unet import Attention_Gate_UNet
from model.channel_unet import ChannelUnet
from model.r2unet import R2U_Net
from model.unet import resnet34_unet
from model.unetpp import NestedUNet
from model.smaat_unet import SmaAt_UNet
# from model.self_attention_unet import get_unet_depthwise_light_encoder_attention_with_skip_connections_decoder
from model.kiunet import kiunet
from model.mobilenetv3_seg import MobileNetV3Seg
from model.res2net import Res2NetSeg

from design.attention_design_one import AttentionDesignOne

# 获得模型实例
def getModel(device, params):
    if params.model == 'UNet':
        model = UNet(3, 1).to(device)
    if params.model == 'resnet34_unet':
        model = resnet34_unet(1, pretrained=False).to(device)
    if params.model == 'unet++':
        params.deepsupervision = True
        model = NestedUNet(params, 3, 1).to(device)
    if params.model == 'Attention_UNet':
        model = Attention_Gate_UNet(3, 1).to(device)
    if params.model == 'segnet':
        model = SegNet(3, 1).to(device)
    if params.model == 'r2unet':
        model = R2U_Net(3, 1).to(device)
    if params.model == 'res2net':
        model = Res2NetSeg([3, 4, 6, 3]).to(device=device)
    if params.model == 'fcn32s':
        model = get_fcn32s(1).to(device)
    if params.model == 'myChannelUnet':
        model = ChannelUnet(3, 1).to(device)
    if params.model == 'fcn8s':
        assert params.dataset != 'esophagus', "fcn8s模型不能用于数据集esophagus，因为esophagus数据集为80x80，经过5次的2倍降采样后剩下2.5x2.5，分辨率不能为小数，建议把数据集resize成更高的分辨率再用于fcn"
        model = get_fcn8s(1).to(device)
    if params.model == 'cenet':
        model = CE_Net_().to(device)
    if params.model == 'smaatunet':
        model = SmaAt_UNet(3, 1).to(device)
    # if params.model == "self_attention_unet":
    #     model = get_unet_depthwise_light_encoder_attention_with_skip_connections_decoder(3,1).to(device)
    if params.model == "kiunet":
        model = kiunet().to(device)
    if params.model == "Lite_RASPP":
        model = MobileNetV3Seg(nclass=1).to(device=device)
    if params.model == "test":
        model = AttentionDesignOne(3, 1).to(device)
    return model