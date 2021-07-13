'''
Author: your name
Date: 2021-07-06 17:00:20
LastEditTime: 2021-07-13 14:27:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /leo/unetzoo/visualizer.py
'''
import argparse
import os
from torchvision import transforms
import torch
from core import getDataset
from model import getModel
from utils.expresult import ExpResult
import numpy as np

import shutil
import os
import torch
import cv2
from torchvision import models
from medcam import medcam
# Grad CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-g', '--gpu', type=str, choices=['0', '1'], default='1')
parser.add_argument('-m', '--model', type=str, default='design_three')
parser.add_argument('-l', '--loss', type=str, choices=['BCE', 'ACELoss', 'hybrid'], default='BCE')
parser.add_argument('-d', '--dataset', type=str,
                    choices=['liver', 'isbicell', 'dsb2018Cell', 'kagglelung', 'driveEye',
                        'esophagus', 'corneal', 'racecar', 'COVID19', 'lung'], default='COVID19')
parser.add_argument('--savedir', default='result/unetzoo', type=str)
parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=3, type=int, metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--deepsupervision', default=0, type=int)
params = parser.parse_args()
print("model : {}".format(params.model))

result = ExpResult(params)
result.expinfo()
shutil.copy('result/save/covid19/20210513-011007-design_three-hybrid-COVID19/train/design_three-COVID19-40-3.pth', result.exp_dir + '/train')

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

# print(model)                    # 显示模型信息
# summary(model, (3, 576, 576))   # 显示模型参数
train_dataloader, val_dataloader, test_dataloader = getDataset(
    params, x_transforms, y_transforms)

print('test start......')
model = result.loadmodel(getModel(device, params))

model = medcam.inject(model, output_dir='attention_maps', backend='gcam',  layer=['attention4', 'attention3', 'attention2', 'attention1', 'conv6'], label='best', save_maps=True)
model.eval()
for pic, _, pic_path, mask_path in test_dataloader:

    pic = pic.to(device)
    predict = model(pic)
    
    if params.deepsupervision:
        predict = torch.squeeze(predict[-1]).cpu().numpy()
    else:
        predict = torch.squeeze(predict).cpu().numpy()
    _,predict = cv2.threshold(predict, 0.1, 1, cv2.THRESH_BINARY)