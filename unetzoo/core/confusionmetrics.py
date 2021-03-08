# -*- coding: utf-8 -*-

#
# Name: Confusion based Metrics for Biomedical Image Segmentation.
# Descrip: 基于混淆矩阵的图像分割的评价度量标准.
# Url: https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
# Author: Li Hao, Cheng Yong
# Notes:
# Date: 2021-03-01
#


import numpy as np

class ConfusionMetrics(object):
    def __init__(self, numClass):
        self.numClass = numClass
        # 每一行之和表示该类别的真实样本数量，每一列之和表示被预测为该类别的样本数量
        self.confmat = np.zeros((self.numClass,) * 2)
        self.smooth = 1e-6

    def set(self, confmat):
        self.confmat = confmat

    # 计算综合准确度
    def accuracy(self):
        # return all class overall pixel accuracy
        # PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confmat).sum() / self.confmat.sum()
        return acc

    # 计算分类准确度
    # https://blog.csdn.net/lingzhou33/article/details/87901365
    def classAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = TP / TP + FP
        classAcc = np.diag(self.confmat) / self.confmat.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    # 计算分类准确度平均值
    # https://blog.csdn.net/lingzhou33/article/details/87901365
    def meanAccuracy(self):
        classAcc = self.classAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96) / 3 = 0.89

    # 计算各类的IoU值
    # https://blog.csdn.net/lingzhou33/article/details/87901365
    def classIoU(self):
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confmat)  # 取对角元素的值，返回列表
        union = np.sum(self.confmat, axis=1) + np.sum(self.confmat, axis=0) - np.diag(
            self.confmat)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / (union + self.smooth)  # 返回列表，其值为各个类别的IoU
        return IoU

    # 计算各类平均的IoU值
    # https://blog.csdn.net/lingzhou33/article/details/87901365
    def meanIoU(self):
        mIoU = np.nanmean(self.classIoU())  # 求各类别IoU的平均
        return mIoU

    # 生成混淆矩阵
    def genConfusionMatrix(self, imgPredict, imgLabel):
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        self.confmat = count.reshape(self.numClass, self.numClass)
        return self.confmat

    # 频权交并比
    def fwIoU(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU = [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = np.sum(self.confmat, axis=1) / np.sum(self.confmat)
        iu = np.diag(self.confmat) / (np.sum(self.confmat, axis=1) + np.sum(self.confmat, axis=0) -
                                      np.diag(self.confmat))
        fwIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confmat += self.genConfusionMatrix(imgPredict, imgLabel)  # 得到混淆矩阵
        return self.confmat

    # 混淆矩阵重置
    def clear(self):
        self.confmat = np.zeros((self.numClass, self.numClass))

    # 查准率(实现两分类情况)
    def precision(self):
        # precision = TP / (TP + FP)
        return self.confmat[1][1] / (self.confmat[1][1] + self.confmat[0][1] + self.smooth)

    # 灵敏性(sensitivity), 查全率(recall), (实现两分类情况)
    def sensitivity(self):
        # recall = TP / (TP + FN)
        return self.confmat[1][1] / (self.confmat[1][1] + self.confmat[1][0])

    # 特异性(实现两分类情况)
    def specificity(self):
        # specificity = TN / (TN + FP)
        return self.confmat[0][0] / (self.confmat[0][0] + self.confmat[0][1])

    # F1综合评分(实现两分类情况)
    def f1_score(self):
        # f1_score = 2 * precision * recall / (precision + recall)
        return 2 * self.precision() * self.sensitivity() / (self.precision() + self.sensitivity() + self.smooth)

    def get_tn(self):
        return self.confmat[0][0]

    def get_fp(self):
        return self.confmat[0][1]

    def get_fn(self):
        return self.confmat[1][0]

    def get_tp(self):
        return self.confmat[1][1]


if __name__ == '__main__':
    pred = np.array([[1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 1]])
    label = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    cma = ConfusionMetrics(2)
    hist = cma.addBatch(pred, label)
    print(hist)
    print('tn is : ', cma.get_tn())
    print('fp is : ', cma.get_fp())
    print('fn is : ', cma.get_fn())
    print('tp is : ', cma.get_tp())
    print('PA is : %f' % cma.accuracy())
    print('cPA is :', cma.classAccuracy())  # 列表
    print('mPA is : %f' % cma.meanAccuracy())
    print('IoU is : ', cma.classIoU())
    print('mIoU is : ', cma.meanIoU())
    print('fwIoU is : ', cma.fwIoU())
    print('accuracy is : ', cma.accuracy())
    print('precision is : ', cma.precision())
    print('recall is : ', cma.recall())
    print('specificity is : ', cma.specificity())
    print('f1_score is : ', cma.f1_score())


# 测试用例
def test_a():
    pred = np.array([[1, 0, 1, 0], [1, 1, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]])
    label = np.array([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0], [0, 1, 0, 0]])
    cma = ConfusionMetrics(2)
    hist = cma.addBatch(pred, label)
    pa = cma.accuracy()
    cpa = cma.classAccuracy()
    mpa = cma.meanAccuracy()
    IoU = cma.classIoU()
    mIoU = cma.meanIoU()
    fwIoU = cma.fwIoU()
    print('-----test_a-----')
    print('hist is :\n', hist)
    print('PA is : %f' % pa)
    print('cPA is :', cpa)  # 列表
    print('mPA is : %f' % mpa)
    print('IoU is : ', IoU)
    print('mIoU is : ', mIoU)
    print('fwIoU is : ', fwIoU)
    print('tn is : ', cma.get_tn())
    print('fp is : ', cma.get_fp())
    print('fn is : ', cma.get_fn())
    print('tp is : ', cma.get_tp())
    print('accuracy is : ', cma.accuracy())
    print('precision is : ', cma.precision())
    print('recall is : ', cma.recall())
    print('specificity is : ', cma.specificity())
    print('f1_score is : ', cma.f1_score())



