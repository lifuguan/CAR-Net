# -*- coding: utf-8 -*-

#
# Title: 测试Liver数据集.
# URL:
# Date: 2020-12-23
#

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from unetzoo.dataset.dataset import LiverDataset


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])
    # Mask只需要转换为tensor
    y_transforms = transforms.ToTensor()

    train_dataset = LiverDataset(r"train", transform=x_transforms, target_transform=y_transforms)
    train_dataloaders = DataLoader(train_dataset, batch_size=8)
    val_dataset = LiverDataset(r"val", transform=x_transforms, target_transform=y_transforms)
    val_dataloaders = DataLoader(val_dataset, batch_size=1)
    test_dataloaders = val_dataloaders

    print(len(train_dataloaders.dataset))

    for x, y, _, mask in train_dataloaders:
        inputs = x
        labels = y
        print(inputs.shape)
        print(labels.shape)
        break

















