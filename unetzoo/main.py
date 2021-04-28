'''
Author: your name
Date: 2021-04-03 22:36:24
LastEditTime: 2021-04-26 22:01:39
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /leo/unetzoo/main.py
'''
# -*- coding: utf-8 -*-

# Author: Li Hao, Cheng Yong
# Date: 2021-03-04 18:15:39
# LastEditTime: 2021-03-05 11:20:03
# LastEditors: Please set LastEditors
# Description: UNET-ZOO训练程序(实验参数JSon形式提供)
# URL: https://github.com/Andy-zhujunwen/UNET-ZOO
# FilePath: \unetzoo\main.py

import argparse
import os
from torch import optim
from torchvision import transforms
import torch
from core import getDataset
from model import getModel
from utils.expresult import ExpResult
from utils.hyperparams import HyperParams
from andytrainer import train
from andytrainer import test


if __name__ == '__main__':
        # 载入参数
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-g', '--gpu', type=str, choices=['0', '1'], default='1')
    parser.add_argument('-m', '--model', type=str, default='UNet')
    parser.add_argument('-l', '--loss', type=str,
                        choices=['BCE', 'ACELoss'], default='BCE')
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['liver', 'isbicell', 'dsb2018Cell', 'kagglelung', 'driveEye', 'esophagus', 'corneal', 'racecar', 'COVID19'], default='COVID19')
    parser.add_argument('--ngpu', default=2, type=int, metavar='G',
                        help='number of gpus to use')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        metavar='N', help='mini-batch size (default: 2)')
    parser.add_argument('--threshold', default='None', type=str)
    parser.add_argument('--action', default='train&test', type=str)
    parser.add_argument('--savedir', default='result/unetzoo', type=str)
    parser.add_argument('--deepsupervision', default=0, type=int)
    params = parser.parse_args()
    print("model : {}".format(params.model))
    if params.threshold == "None":
        params.threshold = None
    else:
        params.threshold = float(params.threshold)
    # 实验结果记录
    result = ExpResult(params)
    result.expinfo()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.empty_cache()

    # 图像预处理
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])
    # mask只需要转换为tensor
    y_transforms = transforms.ToTensor()

    # 准备训练实例,多显卡并行计算
    model = getModel(device, params)

    # print(model)                    # 显示模型信息
    # summary(model, (3, 576, 576))   # 显示模型参数
    train_dataloader, val_dataloader, test_dataloader = getDataset(
        params, x_transforms, y_transforms)
    criterion = torch.nn.BCELoss()    # 损失函数
    optimizer = optim.Adam(model.parameters())

    if 'train' in params.action:
        train(device, params, train_dataloader, val_dataloader,
              model, criterion, optimizer, result)
    if 'test' in params.action:
        test(device, params, test_dataloader, model, result)
