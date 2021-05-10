'''
Author: your name
Date: 2021-04-03 22:36:24
LastEditTime: 2021-05-08 22:01:44
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /leo/unetzoo/core/__init__.py
'''
# -*- coding: utf-8 -*-

#
# Title: generate dataset instance.
# Source:
# Date: 2021-02-26
#

from torch.utils.data import DataLoader
from core.dataset import *


# 获得数据库实例
def getDataset(params, x_transforms, y_transforms):
    train_dataloaders, val_dataloaders, test_dataloaders = None, None, None
    if params.dataset == "liver":
        train_dataset = LiverDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=params.batch_size)
        val_dataset = LiverDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = LiverDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if params.dataset == "esophagus":
        train_dataset = EsophagusDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=params.batch_size)
        val_dataset = EsophagusDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataloaders = val_dataloaders
    if params.dataset == "dsb2018Cell":
        train_dataset = DSB2018CellDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=params.batch_size)
        val_dataset = DSB2018CellDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataloaders = val_dataloaders
    if params.dataset == 'corneal':
        train_dataset = CornealDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=params.batch_size)
        val_dataset = CornealDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = CornealDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if params.dataset == 'driveEye':
        train_dataset = DriveEyeDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=params.batch_size)
        val_dataset = DriveEyeDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = DriveEyeDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if params.dataset == 'isbicell':
        train_dataset = ISBICellDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=params.batch_size)
        val_dataset = ISBICellDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = ISBICellDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if params.dataset == 'kagglelung':
        train_dataset = KaggleLungDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=params.batch_size)
        val_dataset = KaggleLungDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = KaggleLungDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if params.dataset == "racecar":
        train_dataset = RaceCarDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=params.batch_size)
        val_dataset = RaceCarDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataloaders = val_dataloaders
    if params.dataset == "COVID19":
        train_dataset = Covid19Dataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=params.batch_size)
        val_dataset = Covid19Dataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = Covid19Dataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if params.dataset == "lung":
        train_dataset = LungDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=params.batch_size)
        val_dataset = LungDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = LungDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    return train_dataloaders, val_dataloaders, test_dataloaders
