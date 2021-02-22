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
from dataset import *

device = torch.device("cuda")
print("current device is : ", device)

parser = argparse.ArgumentParser(description='PyTorch Liver Training')
parser.add_argument('-m', '--model', type=str, choices=[
                    'atten_gate', 'unet', 'unet_resnet', 'stand_alone_self_attention', 'adr_unet', 'r2_unet', 'segnet'], default='unet_resnet')
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
    model = R2U_Net(1,1)
elif args.model == 'segnet':
    model = SegNet(1,1)
elif args.model == 'stand_alone_self_attention':
    model = get_unet_depthwise_light_encoder_attention_with_skip_connections_decoder(
        1, 1)
elif args.model == 'spatial':
    pass

# 多显卡并行计算
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
model.to(device=device)
model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))

model.train()  # 一定要表明是训练模式!!!
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')


liver_dataset = LiverDataset(
    state='train', scale=501 if args.model == "unet_resnet" else 512)

train_dataloader = DataLoader(
    liver_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)
steps = liver_dataset.__len__() // args.batch_size
print(steps, "steps per epoch")


start = time.time()
train_losses = []
train_ious = []
train_dices = []
train_accuracy = []
train_sensitivity = []
train_precision = []
train_f1_score = []


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
    state = {'net': model.state_dict(
    ), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, model_path)

    # 保存训练损失数据和IoU得分数据
    train_losses_save = np.array(train_losses)
    train_ious_save = np.array(train_ious)
    train_dices_save = np.array(train_dices)
    train_accuracy_save = np.array(train_accuracy)
    train_sensitivity_save = np.array(train_sensitivity)
    train_precision_save = np.array(train_precision)
    train_f1_score_save = np.array(train_f1_score)

    f = open('result/overview.csv', "r+")
    csv_writer = csv.writer(f)
    reader = csv.reader(f)
    original = list(reader)
    # for row in original:
    #     csv_writer.writerow(row)
    msg = [args.model,
           sum([p.data.nelement() for p in model.parameters()]),
           train_losses_save[-1],
           train_ious_save[-1],
           train_dices_save[-1],
           train_accuracy_save[-1],
           train_sensitivity_save[-1],
           train_precision_save[-1],
           train_f1_score_save[-1]]
    csv_writer.writerow(msg)
    f.close()
    print("MSG : ", msg)

    plt.plot(train_losses, label='loss')
    plt.plot(train_ious, label='IoU')
    plt.plot(train_dices, label='Dice')
    plt.plot(train_accuracy, label='Accuracy')
    plt.plot(train_sensitivity, label='Sensitvity')
    plt.plot(train_precision, label='Precision')
    plt.plot(train_f1_score, label='f1_score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title(args.model)
    plt.legend()
    # 保存曲线

    plt.savefig('result/{}/{}/epoch_{}_batch_{}.png'.format(
        args.dataset, args.model, epoch, args.batch_size), bbox_inches='tight')
    plt.show()
    np.save('result/{}/{}/loss_epoch_{}_batch_{}'.format(
        args.dataset, args.model, epoch, args.batch_size), train_losses_save)
    np.save('result/{}/{}/iou_epoch_{}_batch_{}'.format(
        args.dataset, args.model, epoch, args.batch_size), train_ious_save)
    np.save('result/{}/{}/dice_epoch_{}_batch_{}'.format(
        args.dataset, args.model, epoch, args.batch_size), train_dices_save)
    np.save('result/{}/{}/accuracy_epoch_{}_batch_{}'.format(
        args.dataset, args.model, epoch, args.batch_size), train_accuracy_save)
    np.save('result/{}/{}/sensitivity_epoch_{}_batch_{}'.format(
        args.dataset, args.model, epoch, args.batch_size), train_sensitivity_save)
    np.save('result/{}/{}/precision_epoch_{}_batch_{}'.format(
        args.dataset, args.model, epoch, args.batch_size), train_precision_save)
    np.save('result/{}/{}/f1_score_epoch_{}_batch_{}'.format(
        args.dataset, args.model, epoch, args.batch_size), train_f1_score_save)


def train_one_batch(model, x, y, device):
    x, y = x.to(device), y.to(device)

    outputs = model(x)
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.CrossEntropyLoss().cuda()
    loss = loss_fn(outputs, y)
    iou = get_iou_score(outputs, y).mean()
    dice = dice_coef(outputs, y).mean()
    accuracy = get_accuracy(outputs, y)
    sensitivity = get_sensitivity(outputs, y)
    precision = get_precision(outputs, y)

    optimizer.zero_grad()  # 将模型中的梯度设置为0
    loss.backward()
    optimizer.step()
    return loss.item(), iou.item(), dice.item(), accuracy, sensitivity, precision


for epoch in range(1, args.epochs + 1):
    print('-' * 10)
    print('Epoch {}/{}'.format(epoch, args.epochs))
    running_iou = []
    running_loss = []
    running_dice = []
    running_accuracy = []
    running_sensitivity = []
    running_precision = []
    running_f1_score = []
    # 根据batch size提取样本丢进模型里
    for step, (x, y) in enumerate(train_dataloader):
        loss, iou, dice, accuracy, sensitivity, precision = train_one_batch(
            model, x, y, device)
        running_iou.append(iou)
        running_loss.append(loss)
        running_dice.append(dice)
        running_accuracy.append(accuracy)
        running_sensitivity.append(sensitivity)
        running_precision.append(precision)
        running_f1_score.append(2*precision*sensitivity / (precision+sensitivity))
        # print('\r{:6.1f} %\tloss {:8.4f}\tIoU {:8.4f}\tDice {:8.4f}\taccuracy {:8.4f}\tsensitivity {:8.4f}\tprecision {:8.4f}'.format(
        #     100*(step+1)/steps, loss, iou, dice, accuracy, sensitivity, precision), end='')

    print('\r{:6.1f} %\tloss {:8.4f}\tIoU {:8.4f}\tDice {:6.4f}\taccuracy {:6.4f}\tsensitivity {:6.4f}\tprecision {:6.4f}\tf1 score {:6.4f}\t{}'.format(
        100*(step+1)/steps, np.mean(running_loss), np.mean(running_iou), np.mean(running_dice), np.mean(running_accuracy), np.mean(running_sensitivity), np.mean(running_precision), np.mean(running_f1_score), timeSince(start)))
    scheduler.step(np.mean(running_iou))

    train_losses.append(np.mean(running_loss))
    train_ious.append(np.mean(running_iou))
    train_dices.append(np.mean(running_dice))
    train_accuracy.append(np.mean(running_accuracy))
    train_sensitivity.append(np.mean(running_sensitivity))
    train_precision.append(np.mean(running_precision))
    train_f1_score.append(np.mean(running_f1_score))
    if epoch % 50 == 0:
        save_model_args(epoch)
