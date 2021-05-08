'''
Author: your name
Date: 2021-04-09 18:18:40
LastEditTime: 2021-05-08 19:41:22
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /leo/unetzoo/test.py
'''
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
import shutil
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2

from core.confusionmetrics import ConfusionMetrics
from core.file import read_mask, binary_image
from core.segmetrics import get_dice, get_iou, get_hd

# active contour loss function
from core.loss_fn import ACELoss

# training function
from core.metricstable import MetricsTable

if __name__ == '__main__':
        # 载入参数
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-g', '--gpu', type=str, choices=['0', '1'], default='0')
    parser.add_argument('-m', '--model', type=str, default='design_one')
    parser.add_argument('-l', '--loss', type=str,
                        choices=['BCE', 'ACELoss', 'hybrid'], default='hybrid')
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['liver', 'isbicell', 'dsb2018Cell', 'kagglelung', 'driveEye',
                         'esophagus', 'corneal', 'racecar', 'COVID19', 'lung'], default='liver')
    parser.add_argument('--ngpu', default=2, type=int, metavar='G',
                        help='number of gpus to use')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        metavar='N', help='mini-batch size (default: 2)')
    parser.add_argument('--threshold', default='None', type=str)
    parser.add_argument('--action', default='test', type=str)
    parser.add_argument('--savedir', default='result/unetzoo', type=str)
    parser.add_argument('--deepsupervision', default=0, type=int)
    parser.add_argument('--theta', default=0.0005, type=float)
    params = parser.parse_args()
    print("model : {}".format(params.model))
    if params.threshold == "None":
        params.threshold = None
    else:
        params.threshold = float(params.threshold)
    # 实验结果记录
    result = ExpResult(params)
    result.expinfo()
    shutil.copy('result/unetzoo/20210508-191025-design_one-hybrid-liver/train/design_one-liver-40-2.pth', result.exp_dir + '/train')

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
    optimizer = optim.Adam(model.parameters())
    
    print('test start......')
    model = result.loadmodel(model)
    model.eval()
    with torch.no_grad():
        i = 0  # 测试集中第i张图
        cma = ConfusionMetrics(2)
        test_metrics = MetricsTable('test_metrics', result, params)
        num = len(test_dataloader)  # 测试集图片的总数
        for pic, _, pic_path, mask_path in test_dataloader:
            pic = pic.to(device)
            predict = model(pic)
            if params.deepsupervision:
                predict = torch.squeeze(predict[-1]).cpu().numpy()
            else:
                predict = torch.squeeze(predict).cpu().numpy()
            _,predict = cv2.threshold(predict, 0.1, 1, cv2.THRESH_BINARY)
            # 图像二值化处理
            mask_img= read_mask(mask_path[0], np.size(predict,1))
            image_mask = binary_image(mask_img, 30)  # check it 
            image_mask = binary_image(image_mask, 0.5)
            # 计算度量值
            cma.genConfusionMatrix(predict.astype(np.int32), image_mask.astype(np.int32))
            accuracy = cma.accuracy()
            precision = cma.precision()
            sensitivity = cma.sensitivity()
            specificity = cma.specificity()
            f1score = cma.f1_score()
            meanIoU = cma.meanIoU()
            fwIoU = cma.fwIoU()
            iou = get_iou(predict, image_mask)
            dice = get_dice(predict, image_mask)
            hd = get_hd(predict, image_mask)
            test_metrics.addmetric(accuracy, precision, sensitivity, specificity, f1score, meanIoU, fwIoU, iou, dice, hd)

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('input')
            origin_img = Image.open(pic_path[0])
            plt.imshow(origin_img)
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title('predict')
            plt.imshow(predict, cmap='Greys_r')
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('mask')
            mask_img = Image.open(mask_path[0])
            plt.imshow(mask_img, cmap='Greys_r')

            if params.dataset == 'driveEye':
                saved_predict = result.test_dir + '/' + mask_path[0].split('/')[-1]
                saved_predict = saved_predict.split('.')[0] + '.tif'
                plt.savefig(saved_predict)
            else:
                plt.savefig(result.test_dir + '/' + mask_path[0].split('/')[-1][:-4] + "jpg")

            result.print('accuracy = {}'.format(accuracy))
            result.print('precision = {}'.format(precision))
            result.print('sensitivity = {}'.format(sensitivity))
            result.print('specificity = {}'.format(specificity))
            result.print('f1score = {}'.format(f1score))
            result.print('meanIoU = {}'.format(meanIoU))
            result.print('fwIoU = {}'.format(fwIoU))
            result.print('iou = {}'.format(iou))
            result.print('dice = {}'.format(dice))
            print('hd = {}'.format(hd))
        if i < num:
                i += 1  # 处理验证集下一张图
        test_metrics.print()