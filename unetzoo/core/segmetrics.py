# -*- coding: utf-8 -*-

#
# Name: Metrics for Image Segmentation.
# Author: Li Hao, Cheng Yong
# Title:
# Notes: 图像分割的评价度量标准.
# Date: 2021-03-01
#


import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


# 查准率指标(precision)
def precision(outputs, labels, threshold = 0.5):
    '''
    计算查准率
    :param outputs: output image
    :param labels: label image
    :param threshold: default 0.5
    :return:
    '''
    smooth = 1e-6
    if not torch.is_tensor(outputs):
        outputs = torch.from_numpy(outputs)
    outputs = outputs >= threshold
    if not torch.is_tensor(labels):
        labels = torch.from_numpy(labels)
    labels = labels == torch.max(labels)
    # TP : True Positive
    # FP : False Positive
    TP = ((outputs == 1).byte() + (labels == 1).byte()) == 2
    FP = ((outputs == 1).byte() + (labels == 0).byte()) == 2
    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + smooth)
    # print("precision: ", (outputs==1) + (labels==0), " ", torch.sum(TP+FP), "\n")
    return PC


# 准确度指标(accuracy)
def accuracy(outputs, labels, threshold = 0.5):
    '''
    :param outputs: output image
    :param labels: label image
    :param threshold: defautl 0.5
    :return:
    '''
    smooth = 1e-6
    if not torch.is_tensor(outputs):
        outputs = torch.from_numpy(outputs)
    outputs = outputs >= threshold
    if not torch.is_tensor(labels):
        labels = torch.from_numpy(labels)
    labels = labels == torch.max(labels)
    corr = torch.sum(outputs == labels)
    tensor_size = outputs.size(0) * outputs.size(1)
    AC = float(corr) / float(tensor_size + smooth)
    # print(corr, tensor_size)
    return AC

# Sensitivity(recall)
def sensitivity(outputs, labels, threshold = 0.5):
    '''
    :param outputs: output image
    :param labels: label image
    :param threshold: defautl 0.5
    :return:
    '''
    smooth = 1e-6
    if not torch.is_tensor(outputs):
        outputs = torch.from_numpy(outputs)
    outputs = outputs >= threshold
    if not torch.is_tensor(labels):
        labels = torch.from_numpy(labels)
    # TP : True Positive
    # FN : False Negative
    TP = ((outputs == 1).byte() + (labels == 1).byte()) == 2
    FN = ((outputs == 0).byte() + (labels == 1).byte()) == 2
    # print(TP.size(), FN.size())
    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + smooth)
    return SE

# Hausdorff值
# https://github.com/Andy-zhujunwen/UNET-ZOO
def get_hd(outputs, labels):
    '''
    :param outputs: numpy array
    :param labels: numpy array
    :param threshold: defautl 0.5
    :return:
    '''
    hd1 = directed_hausdorff(outputs, labels)[0]
    hd2 = directed_hausdorff(outputs, labels)[0]
    return max(hd1, hd2)

# Hausdorff值
# https://github.com/Andy-zhujunwen/UNET-ZOO
def hd(outputs, labels, threshold = 0.5):
    '''
    :param outputs: numpy array
    :param labels: numpy array
    :param threshold: defautl 0.5
    :return:
    '''
    outputs = outputs >= threshold
    if torch.is_tensor(outputs):
        outputs = outputs.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    hd1 = directed_hausdorff(outputs, labels)[0]
    hd2 = directed_hausdorff(outputs, labels)[0]
    return max(hd1, hd2)

# 计算iou得分
# https://github.com/Andy-zhujunwen/UNET-ZOO
def get_iou(outputs, labels):
    '''
    :param outputs: numpy array
    :param labels: numpy array
    :return:
    '''
    predict = outputs.astype(np.int16)
    interArea = np.multiply(predict, labels)
    tem = predict + labels
    unionArea = tem - interArea
    inter = np.sum(interArea)
    union = np.sum(unionArea)
    iou_tem = inter / union
    return iou_tem


# https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
def iou(outputs, labels, threshold = 0.5):
    '''
    :param outputs: H x W
    :param labels: H x W
    :return:
    '''
    smooth = 1e-5
    outputs = outputs > threshold
    if torch.is_tensor(outputs):
        outputs = outputs.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()
    iou = (intersection + smooth) / (union + smooth)
    return iou

# https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
def batch_iou(outputs: torch.Tensor, labels: torch.Tensor):
    smooth = 1e-5
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0
    iou = (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0
    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    return iou  # thresholded  or thresholded.mean() if you are interested in average across the batch

# https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
def batch_iou_numpy(outputs: np.array, labels: np.array):
    '''
    :param outputs: BATCH x 1 x H x W
    :param labels: BATCH x H x W
    :return:
    '''
    smooth = 1e-5
    outputs = outputs.squeeze(1)
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    iou = (intersection + smooth) / (union + smooth)
    # thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    return iou  # thresholded or thresholded.mean()


# Source: https://github.com/Andy-zhujunwen/UNET-ZOO
def get_dice(outputs, labels):
    outputs = outputs.astype(np.int16)
    intersection = (outputs * labels).sum()
    dice = (2. * intersection) / (outputs.sum() + labels.sum())
    return dice

# https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/metrics.py
def dice(outputs, labels, threshold = 0.5):
    smooth = 1e-6  # 防止0除
    outputs = outputs > threshold
    if torch.is_tensor(outputs):
        outputs = outputs.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    intersection = (outputs * labels).sum()
    dice = (2. * intersection + smooth) / (outputs.sum() + labels.sum() + smooth)
    return dice


# =================
# 测试用例
# =================
# precision
def test_precision():
    outputs = np.array([[0.65, 0.56, 0.84, 0.67], [0.64, 0.24, 0.23, 0.16], [0.18, 0.23, 0.56, 0.67], [0.67, 0.34, 0.83, 0.98]])
    labels = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    print(precision(outputs, labels))   # 0.499999950000005
    outputs_tensor = torch.from_numpy(outputs)
    labels_tensor = torch.from_numpy(labels)
    print(precision(outputs_tensor, labels_tensor))   # 0.499999950000005

# 测试用例accuracy
def test_accuracy():
    outputs = np.array(
        [[0.65, 0.56, 0.84, 0.67], [0.64, 0.24, 0.23, 0.16], [0.18, 0.23, 0.56, 0.67], [0.67, 0.34, 0.83, 0.98]])
    outputs_tensor = torch.from_numpy(outputs)
    labels = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    labels_tensor = torch.from_numpy(labels)
    print(accuracy(outputs, labels))      # 0.5624999648437522
    print(accuracy(outputs_tensor, labels_tensor))      # 0.5624999648437522

# 测试用例sensitivity
def test_sensitivity():
    outputs = np.array([[0.65, 0.56, 0.84, 0.67], [0.64, 0.24, 0.23, 0.16], [0.18, 0.23, 0.56, 0.67], [0.67, 0.34, 0.83, 0.98]])
    outputs_tensor = torch.from_numpy(outputs)
    labels = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    labels_tensor = torch.from_numpy(labels)
    print(sensitivity(outputs, labels))     # 0.7142856122449125
    print(sensitivity(outputs_tensor, labels_tensor))   # 0.7142856122449125

# 测试用例hd
def test_hd():
    outputs = np.array(
        [[0.65, 0.56, 0.84, 0.67], [0.64, 0.24, 0.23, 0.16], [0.18, 0.23, 0.56, 0.67], [0.67, 0.34, 0.83, 0.98]])
    outputs_tensor = torch.from_numpy(outputs)
    labels = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    labels_tensor = torch.from_numpy(labels)
    print(hd(outputs, labels))  # 1.4142135623730951
    print(hd(outputs_tensor, labels_tensor))  # 1.4142135623730951

def test_get_iou():
    outputs = np.array([[1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 1]])
    labels = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    print(get_iou(outputs, labels))   # 0.4166666666666667

def test_iou():
    outputs = np.array(
        [[0.65, 0.56, 0.84, 0.67], [0.64, 0.24, 0.23, 0.16], [0.18, 0.23, 0.56, 0.67], [0.67, 0.34, 0.83, 0.98]])
    outputs_tensor = torch.from_numpy(outputs)
    labels = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    labels_tensor = torch.from_numpy(labels)
    print(iou(outputs, labels))                # 0.41666715277737265
    print(iou(outputs_tensor, labels_tensor))  # 0.41666715277737265

def test_get_dice():
    outputs = np.array([[1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 1]])
    labels = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    print(get_dice(outputs, labels))    # 0.5882352941176471

def test_dice():
    outputs = np.array(
        [[0.65, 0.56, 0.84, 0.67], [0.64, 0.24, 0.23, 0.16], [0.18, 0.23, 0.56, 0.67], [0.67, 0.34, 0.83, 0.98]])
    outputs_tensor = torch.from_numpy(outputs)
    labels = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    labels_tensor = torch.from_numpy(labels)
    print(dice(outputs, labels))                # 0.5882353183390988
    print(dice(outputs_tensor, labels_tensor))  # 0.5882353183390988


# 综合测试用例a
def test_a():
    pred = np.array([[1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 1]])
    label = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    pred_tensor = torch.from_numpy(pred)
    label_tensor = torch.from_numpy(label)
    print("sensitivity_torch: ", sensitivity(pred_tensor, label_tensor))  # 0.7142856122449125
    print("precision_torch: ", precision(pred_tensor, label_tensor))  # 0.499999950000005
    print("accuracy: ", accuracy(pred_tensor, label_tensor))  # 0.5624999648437522


# 综合测试用例b
def test_b():
    pred = np.array([[1, 0, 1, 0], [1, 1, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]])
    label = np.array([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0], [0, 1, 0, 0]])
    pred_tensor = torch.from_numpy(pred)
    label_tensor = torch.from_numpy(label)
    print("-----" * 5)
    print("iou numpy:", iou(pred, label))     # 0.6666669444442129
    print("iou torch:", iou(pred_tensor, label_tensor))   # 0.6666669444442129
    print("get_iou:", get_iou(pred, label))    # 0.6666666666666666

    print("-----" * 5)
    print('get_dice is : %f' % get_dice(pred, label))   # 0.800000
    print("dice_numpy:", dice(pred, label))    # 0.8000000099999995
    print("dice_torch:", dice(pred, label))    # 0.8000000099999995

    print("-----" * 5)
    pred = np.array([[1, 0, 1, 0], [1, 1, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]])
    label = np.array([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0], [0, 1, 0, 0]])
    print("sensitivity numpy: ", sensitivity(pred, label))  # 0.7272726611570308
    print("precision numpy: ", precision(pred, label))  # .8888887901234679
    print("accuracy numpy: ", accuracy(pred, label))  # 0.7499999531250029


if __name__ == '__main__':
    test_precision()
    test_accuracy()
    test_sensitivity()
    test_hd()
    test_get_iou()
    test_iou()
    test_get_dice()
    test_dice()
    test_a()
    test_b()

