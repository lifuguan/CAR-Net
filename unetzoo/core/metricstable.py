# -*- coding: utf-8 -*-

#
# Name: Metrics Manager for Image Segmentation.
# Author: Cheng Yong
# Title:
# Notes: 管理图像分割度量的十个指标值.
# Date: 2021-03-04
#


import numpy as np
from common.utils.expresult import ExpResult
from common.utils.hyperparams import HyperParams


class MetricsTable(object):
    def __init__(self, name, result):
        self.name = name
        self.result = result
        self.len = 0
        self.accuracy_list = []
        self.precision_list = []
        self.sensitivity_list = []
        self.specificity_list = []
        self.f1_score_list = []
        self.meaniou_list = []
        self.fwiou_list = []
        self.iou_list = []
        self.dice_list = []
        self.hd_list = []

    def addmetric(self, accuracy, precision, sensitivity, specificity, f1_score, meaniou, fwiou, iou, dice, hd):
        self.accuracy_list.append(accuracy)
        self.precision_list.append(precision)
        self.sensitivity_list.append(sensitivity)
        self.specificity_list.append(specificity)
        self.f1_score_list.append(f1_score)
        self.meaniou_list.append(meaniou)
        self.fwiou_list.append(fwiou)
        self.iou_list.append(iou)
        self.dice_list.append(dice)
        self.hd_list.append(hd)
        self.len += 1

    def savemetrics(self, phase):
        self.result.savemetrics(phase, 'accuracy-precision-sensitivity-specificity-f1_score-meaniou-fwiou-iou-dice-hd',
                                self.accuracy_list, self.precision_list, self.sensitivity_list, self.specificity_list,
                                self.f1_score_list, self.meaniou_list, self.fwiou_list, self.iou_list, self.dice_list,
                                self.hd_list)

    def metrics_mean(self):
        return sum(self.accuracy_list) / self.len, sum(self.precision_list) / self.len, sum(
            self.sensitivity_list) / self.len, \
               sum(self.specificity_list) / self.len, sum(self.f1_score_list) / self.len, sum(
            self.meaniou_list) / self.len, \
               sum(self.fwiou_list) / self.len, sum(self.iou_list) / self.len, sum(self.dice_list) / self.len, \
               sum(self.hd_list) / self.len

    def metrics_total(self):
        return sum(self.accuracy_list), sum(self.precision_list), sum(self.sensitivity_list), sum(
            self.specificity_list), \
               sum(self.f1_score_list), sum(self.meaniou_list), sum(self.fwiou_list), sum(self.iou_list), \
               sum(self.dice_list), sum(self.hd_list)

    # 计算平均iou
    def avg_iou(self):
        return sum(self.iou_list) / self.len

    def print(self):
        self.result.print(self.name)
        self.result.print('avg_accuracy = %f' % (sum(self.accuracy_list) / self.len))
        self.result.print('avg_precision = %f' % (sum(self.precision_list) / self.len))
        self.result.print('avg_sensitivity = %f' % (sum(self.sensitivity_list) / self.len))
        self.result.print('avg_specificity = %f' % (sum(self.specificity_list) / self.len))
        self.result.print('avg_f1_score = %f' % (sum(self.f1_score_list) / self.len))
        self.result.print('avg_meaniou = %f' % (sum(self.meaniou_list) / self.len))
        self.result.print('avg_fwiou = %f' % (sum(self.fwiou_list) / self.len))
        self.result.print('avg_iou = %f' % (sum(self.iou_list) / self.len))
        self.result.print('avg_dice = %f' % (sum(self.dice_list) / self.len))
        self.result.print('avg_hd = %f' % (sum(self.hd_list) / self.len))


if __name__ == '__main__':
    params = HyperParams("E:/studio/learn/python/src/lab/unetzoo/config/unet.json")
    result = ExpResult(params)
    table = MetricsTable('test', result)

    for i in range(13):
        value = np.random.rand(10)
        table.addmetric(value[0],value[1],value[2],value[3],value[4],value[5],value[6],value[7],value[8],value[9])

    print(table.metrics_mean())
    print(table.metrics_total())
    table.print()
    table.savemetrics('train')


