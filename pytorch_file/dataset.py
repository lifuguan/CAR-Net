import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
'''
Liver Datasets Loader
'''

class LiverDataset(Dataset):
    def __init__(self, state, scale = 512):
        self.state = state
        self.train_root = r"dataset/liver/train"
        self.val_root = r"dataset/liver/val"

        # 若assert为异常值，则报错
        assert self.state == 'train' or self.state == 'val'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root

        pics = []
        masks = []
        n = len(os.listdir(root)) // 2  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        for i in range(n):
            pics.append(os.path.join(root, "%03d.png" % i))  # liver is %03d
            masks.append(os.path.join(root, "%03d_mask.png" % i))
        self.pics, self.masks = pics, masks

        self.transform = T.Compose(
            [T.Grayscale(), T.Resize((scale, scale), interpolation=Image.NEAREST), T.ToTensor()])

    def __getitem__(self, index):
        origin_x = Image.open(self.pics[index])
        origin_y = Image.open(self.masks[index])
        return self.transform(origin_x), self.transform(origin_y)

    def __len__(self):
        return len(self.pics)


