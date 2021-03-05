from torchsummary import summary
import numpy as np
import os
import argparse
import time
import math
import glob
import pandas as pd
import matplotlib.pyplot as plt
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision


from eval import *
from MODELS.attention_gate_unet import AttenGateUNet
from MODELS.unet import UNet
from MODELS.self_attention_unet import *
from MODELS.unet_resnet import UnetResnet
from MODELS.adr_unet import AdrUNet
from MODELS.r2unet import R2U_Net
from MODELS.segnet import SegNet
from MODELS.smaat_unet import SmaAt_UNet
from MODELS.unetpp import NestedUNet
from dataset import *

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def save_model_args(epoch):
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    if not os.path.exists('result/{}/{}'.format(args.dataset, args.model)):
        os.mkdir('result/{}/{}'.format(args.dataset, args.model))
    # 模型地址
    model_path = 'result/{}/{}/epoch_{}_batch_{}_{}.pkl'.format(
        args.dataset, args.model, epoch, args.batch_size, 4)
    # 三个参数：网络参数；优化器参数；epoch
    state = {'net': model.state_dict()}
    torch.save(state, model_path)

    # 保存训练损失数据和IoU得分数据
    f = open('result/overview.csv', "r+")
    csv_writer = csv.writer(f)
    reader = csv.reader(f)
    original = list(reader)
    # for row in original:
    #     csv_writer.writerow(row)
    msg = [args.model,
           sum([p.data.nelement() for p in model.parameters()]),
        #    train_losses_save[-1],
           running_iou[-1],
           running_dice[-1],
           running_precision[-1],
        #    train_sensitivity_save[-1],
           running_precision[-1],
           running_f1_score[-1]]
    csv_writer.writerow(msg)
    f.close()
    print("MSG : ", msg)

    # plt.plot(train_losses, label='loss')
    plt.plot(running_iou, label='IoU')
    plt.plot(running_dice, label='Dice')
    plt.plot(running_accuracy, label='Accuracy')
    # plt.plot(train_sensitivity, label='Sensitvity')
    plt.plot(running_precision, label='Precision')
    plt.plot(running_f1_score, label='f1_score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title(args.model)
    plt.legend()
    # 保存曲线

    plt.savefig('result/{}/{}/epoch_{}_batch_{}.png'.format(
        args.dataset, args.model, epoch, args.batch_size), bbox_inches='tight')
    plt.show()
    # np.save('result/{}/{}/loss_epoch_{}_batch_{}'.format(
    #     args.dataset, args.model, epoch, args.batch_size), train_losses_save)
    np.save('result/{}/{}/iou_epoch_{}_batch_{}'.format(
        args.dataset, args.model, epoch, args.batch_size), running_iou)
    np.save('result/{}/{}/dice_epoch_{}_batch_{}'.format(
        args.dataset, args.model, epoch, args.batch_size), running_dice)
    np.save('result/{}/{}/accuracy_epoch_{}_batch_{}'.format(
        args.dataset, args.model, epoch, args.batch_size), running_accuracy)
    # np.save('result/{}/{}/sensitivity_epoch_{}_batch_{}'.format(
    #     args.dataset, args.model, epoch, args.batch_size), train_sensitivity_save)
    np.save('result/{}/{}/precision_epoch_{}_batch_{}'.format(
        args.dataset, args.model, epoch, args.batch_size), running_precision)
    np.save('result/{}/{}/f1_score_epoch_{}_batch_{}'.format(
        args.dataset, args.model, epoch, args.batch_size), running_f1_score)

def modelTraining():
    model.train()  # 一定要表明是训练模式!!!

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')


    liver_dataset = LiverDataset(
        state='train', scale=501 if args.model == "unet_resnet" else 512)

    train_dataloader = DataLoader(
        liver_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)
    steps = liver_dataset.__len__() // args.batch_size
    print(steps, "steps per epoch")
    metric = Evaluator(2)
    start = time.time()
    training_iou = []
    training_loss = []
    for epoch in range(1, args.epochs + 1):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, args.epochs))
        metric.clear()
        running_iou = []
        running_loss = []
        # 根据batch size提取样本丢进模型里
        for step, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(outputs, y)
            pred = outputs.data.cpu().numpy()
            label = y.cpu().numpy()
            metric.addBatch(pred, label)
            iou = metric.meanIoU()
            optimizer.zero_grad()  # 将模型中的梯度设置为0
            loss.backward()
            optimizer.step()
            training_iou.append(iou)
            training_loss.append(loss.item())
        print('\r{:6.1f} %\tloss {:8.4f}\tIoU {:8.4f}\t{}'.format(
            100*(step+1)/steps, np.mean(training_loss), np.mean(training_iou), timeSince(start)))
        scheduler.step(np.mean(training_iou))

running_iou = []
running_dice = []
running_accuracy = []
running_precision = []
running_f1_score = []

def modelValidating():
    model.eval()  # 一定要表明是评估模式!!!

    liver_dataset = LiverDataset(
        state='val', scale=501 if args.model == "unet_resnet" else 512)

    train_dataloader = DataLoader(
        liver_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)

    start = time.time()
    metric = Evaluator(2)  # 3表示有3个分类，有几个分类就填几
    metric.clear()

    # 根据batch size提取样本丢进模型里
    for step, (x, y) in enumerate(train_dataloader):
        x, y = x.cuda(), y.cuda()
        
        with torch.no_grad():
            outputs = model(x)
        
        pred = outputs.data.cpu().numpy()
        label = y.cpu().numpy()
        metric.addBatch(pred, label)

        iou = metric.meanIoU()
        dice = dice_coef(outputs, y)
        accuracy = metric.meanAccuracy()
        f1_score = metric.f1_score()
        precision = metric.precision()

        running_iou.append(iou)
        running_dice.append(dice)
        running_accuracy.append(accuracy)
        running_precision.append(precision)
        running_f1_score.append(f1_score)

    print('Validation result:\tIoU {:8.4f}\tDice {:6.4f}\taccuracy {:6.4f}\tprecision {:6.4f}\tf1 score {:6.4f}\t{}'.format(
    np.mean(running_iou), np.mean(running_dice), np.mean(running_accuracy), np.mean(running_precision), np.mean(running_f1_score), timeSince(start)))

    save_model_args(50)


if __name__ == '__main__':
    device = torch.device("cuda")
    print("current device is : ", device)

    parser = argparse.ArgumentParser(description='PyTorch Liver Training')
    parser.add_argument('-m', '--model', type=str, choices=[
                        'atten_gate', 'unet', 'unet_resnet', 'stand_alone_self_attention',
                        'adr_unet', 'r2_unet', 'segnet', 'smaat_unet', 'unet++'], default='atten_gate')
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['liver', 'drive'], default='liver')
    parser.add_argument('--ngpu', default=2, type=int, metavar='G',
                        help='number of gpus to use')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        metavar='N', help='mini-batch size (default: 2)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')

    args = parser.parse_args()
    print('=' * 10)
    print("Trainning model: {}; dataset:{}".format(args.model, args.dataset))
    print('=' * 10)

    if args.model == 'atten_gate':
        model = AttenGateUNet()
    elif args.model == 'unet':
        model = UNet(1, 1)
    elif args.model == 'unet_resnet':
        model = UnetResnet()
    elif args.model == 'adr_unet':
        model = AdrUNet()
    elif args.model == 'r2_unet':
        model = R2U_Net(1, 1)
    elif args.model == 'smaat_unet':
        model = SmaAt_UNet(1, 1)
    elif args.model == 'segnet':
        model = SegNet(1, 1)
    elif args.model == 'unet++':
        model = NestedUNet(False, 1, 1)
    elif args.model == 'stand_alone_self_attention':
        model = get_unet_depthwise_light_encoder_attention_with_skip_connections_decoder(
            1, 1)
    elif args.model == 'spatial':
        pass

    # 多显卡并行计算
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model.to(device=device)
    # model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))

    modelTraining()

    modelValidating()