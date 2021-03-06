'''
Author: your name
Date: 2021-04-03 22:36:24
LastEditTime: 2021-04-26 22:15:42
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /leo/unetzoo/core/file.py
'''
import cv2
import imageio
import numpy as np

# 读取mask文件
# https://github.com/Andy-zhujunwen/UNET-ZOO
def read_mask(mask_name, size):
    image_mask = cv2.imread(mask_name, 0)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
    image_mask = cv2.resize(image_mask, (size, size))
    return image_mask

# 创建二进制图像文件
# https://github.com/Andy-zhujunwen/UNET-ZOO
def binary_image(image, threshold):
    height = image.shape[0]
    weight = image.shape[1]
    for row in range(height):
        for col in range(weight):
            if image[row, col] < threshold:  # 由于输出的predit是0~1范围的，其中值越靠近1越被网络认为是肝脏目标，所以取0.5为阈值
                image[row, col] = 0
            else:
                image[row, col] = 1
    return image