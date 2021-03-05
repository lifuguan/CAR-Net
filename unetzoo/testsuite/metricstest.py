# -*- coding: utf-8 -*-

#
# Title: 测试segmetrics类
# Arxiv:
# Source:
# Date: 2021-03-01
#

import numpy as np
import torch
from common.metrics.segmetrics import iou
from unetzoo.core.metrics import get_hd, get_dice, get_iou, read_mask, binary_image


if __name__ == '__main__':
    img_y = np.loadtxt(open('D:/download/img_y.csv', "rb"), delimiter=",", skiprows=0)
    mask_name = 'D:/dataset/biomed/UNetZoo/isbi/train/label/19.tif'
    print(get_iou(mask_name, img_y))    # 0.8049783925404531
    print(get_dice(mask_name, img_y))   # 0.8919534946980392
    print(get_hd(mask_name, img_y))     # 13.674794331177344

