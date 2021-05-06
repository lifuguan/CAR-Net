# -*- coding: utf-8 -*-

#
# Title: 训练程序
# Modified: Cheng Yong
# URL: https://github.com/Andy-zhujunwen/UNET-ZOO
# Date: 2021-01-24
#

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
from core.aceloss import ACELoss

# training function
from core.metricstable import MetricsTable


def train(device, params, train_dataloader, val_dataloader, model, criterion, optimizer, result, vis):
    result.print('train start......')
    best_iou, aver_iou, aver_dice, aver_hd = 0, 0, 0, 0
    num_epochs = params.epochs
    threshold = params.threshold
    loss_list = []
    epoch_metrics = MetricsTable('epoch_metrics', result, params)

    for epoch in range(num_epochs):
        # 标明是训练模式
        model = model.train()
        
        result.print('Epoch {} / {}'.format(epoch, num_epochs - 1))
        result.print('-' * 15)
        train_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        with tqdm(total=train_size // params.batch_size) as pbar:
            for x, y, _, mask in train_dataloader:
                step += 1
                inputs = x.to(device)
                labels = y.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                if params.deepsupervision:
                    outputs = model(inputs)
                    loss = 0
                    for output in outputs:
                        loss += criterion(output, labels)
                    loss /= len(outputs)
                else:
                    output = model(inputs)
                    if params.loss == "BCE":
                        loss = criterion(output, labels)
                    elif params.loss == "hybrid":
                        loss = criterion(output, labels) + params.theta * ACELoss(output, labels)
                    elif params.loss == "ACELoss":
                        loss = ACELoss(output, labels)

                if threshold != None:
                    if loss > threshold:
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                else:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                pbar.set_description('epoch %d train_loss: %0.3f' % (epoch, loss.item()))
                pbar.update(1)
                result.print("%d / %d, train_loss: %0.3f" % (step, (train_size - 1) // train_dataloader.batch_size + 1, loss.item()))

        loss_list.append(epoch_loss)
        best_iou, (avg_accuracy, avg_precision, avg_sensitivity, avg_specificity, avg_f1score, avg_meanIoU, avg_fwIoU, avg_iou, \
        avg_dice, avg_hd) = val(device, params, model, best_iou, val_dataloader, result, vis)
        epoch_metrics.addmetric(avg_accuracy, avg_precision, avg_sensitivity, avg_specificity, avg_f1score, \
                                avg_meanIoU, avg_fwIoU, avg_iou, avg_dice, avg_hd)
        result.print("epoch %d loss: %0.3f" % (epoch, epoch_loss))
        vis.plot_many_stack({'iou':avg_iou, 'dice':avg_dice, 'f1_score':avg_f1score})
    result.savelosses('loss', loss_list)
    epoch_metrics.savemetrics('train')
    return model


# validation
def val(device, params, model, best_iou, val_dataloader, result, vis):
    model = model.eval()
    with torch.no_grad():
        i = 0  # 验证集中第i张图
        cma = ConfusionMetrics(2)
        val_metrics = MetricsTable('val_metrics', result, params)
        num = len(val_dataloader)  # 验证集图片的总数
        for x, _, pic, mask in val_dataloader:
            x = x.to(device)
            y = model(x)
            if params.deepsupervision:
                img_y = torch.squeeze(y[-1]).cpu().numpy()
                img_x = torch.squeeze(x[-1]).cpu().permute(1,2,0).numpy()
            else:
                img_y = torch.squeeze(y).cpu().numpy()
                img_x = torch.squeeze(x).cpu().permute(1,2,0).numpy()

            # 图像二值化处理
            mask_img= read_mask(mask[0], np.size(img_y,1))
            image_mask = binary_image(mask_img, 30)  # check it 
            img_y = binary_image(img_y, 0.5)

            if i == 5:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 3, 1)
                ax1.set_title('Origin')
                plt.imshow(img_x, cmap = 'bone')
                ax2 = fig.add_subplot(1, 3, 2)
                ax2.set_title('GT')
                plt.imshow(img_x, cmap = 'bone')
                plt.imshow(image_mask, alpha = 0.5, cmap = 'nipy_spectral')
                ax3 = fig.add_subplot(1, 3, 3)
                ax3.set_title('Predict')
                plt.imshow(img_x, cmap = 'bone')
                plt.imshow(img_y, alpha = 0.5, cmap = 'nipy_spectral')
                assert vis.vis.check_connection()
                vis.vis.matplot(plt)
                
                plt.close()
            # 计算度量值
            cma.genConfusionMatrix(img_y.astype(np.int32), image_mask.astype(np.int32))
            accuracy = cma.accuracy()
            precision = cma.precision()
            sensitivity = cma.sensitivity()
            specificity = cma.specificity()
            f1score = cma.f1_score()
            meanIoU = cma.meanIoU()
            fwIoU = cma.fwIoU()
            iou = get_iou(img_y, image_mask)
            dice = get_dice(img_y, image_mask)
            hd = get_hd(img_y, image_mask)
            val_metrics.addmetric(accuracy, precision, sensitivity, specificity, f1score, meanIoU, fwIoU, iou, dice, hd)
            if i < num:
                i += 1  # 处理验证集下一张图
        val_metrics.print()
        aver_iou = val_metrics.avg_iou()
        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou, best_iou))
            best_iou = aver_iou
            print('=======>save best model!')
            result.savemodel(model)
        return best_iou, val_metrics.metrics_mean()


# test
def test(device, params, test_dataloader, model, result):
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


