import torch
import torch.nn as nn
import torch.optim
import numpy as np

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)
        self.smooth = 1e-6
 
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc
 
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
 
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc) # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89
 
    def meanIntersectionOverUnion(self):
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix) # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix) # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表 
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU) # 求各类别IoU的平均
        return mIoU

    def intersectionOverUnion(self):
        # IoU = TP / (TP + FP + FN)
        intersection = 2 * np.diag(self.confusionMatrix)[0] # 取对角元素的值中的第一个（为TP）
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix) # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表 
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def diceCofficient(self):
        # IoU = 2 * TP / (2 * TP + FP + FN)
        intersection = 2 * np.diag(self.confusionMatrix)[0] # 取对角元素的值中的第一个（为TP）
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表 
        dice = intersection / union  # 其值为dice
        return dice

    def genConfusionMatrix(self, imgPredict, imgLabel): # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return np.flip(confusionMatrix)
 
    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
 
 
    def addBatch(self, outputs, labels):
        assert outputs.shape == labels.shape
        if torch.is_tensor(outputs):
            outputs = torch.sigmoid(outputs).data.cpu().numpy().astype(np.int)
        if torch.is_tensor(labels):
            labels = labels.data.cpu().numpy().astype(np.int)
        self.confusionMatrix += self.genConfusionMatrix(outputs, labels)
 
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
 


# IOU指标
def get_iou_score(outputs, labels):
    A = labels.squeeze().bool()
    pred = torch.where(outputs < 0., torch.zeros_like(
        outputs), torch.ones_like(outputs))
    B = pred.squeeze().bool()
    intersection = (A & B).float().sum((1, 2))
    union = (A | B).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou

# DICE指标
def dice_coef(outputs, labels):#output为预测结果 target为真实结果
    smooth = 1e-6 #防止0除
    if torch.is_tensor(outputs):
        outputs = torch.sigmoid(outputs).data.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.data.cpu().numpy()

    intersection = (outputs * labels).sum()
    dice = (2. * intersection + smooth) / (outputs.sum() + labels.sum() + smooth)

    return dice

# precision 和 recall 指标
def precision_and_recall(label_gt, label_pred, n_class):
    from sklearn.metrics import precision_score, recall_score
    assert len(label_gt) == len(label_pred)
    precision = np.zeros(n_class, dtype=np.float32)
    recall = np.zeros(n_class, dtype=np.float32)
    img_A = np.array(label_gt, dtype=np.float32).flatten()
    img_B = np.array(label_pred, dtype=np.float32).flatten()
    precision[:] = precision_score(img_A, img_B, average=None, labels=range(n_class))
    recall[:] = recall_score(img_A, img_B, average=None, labels=range(n_class))

    return precision, recall

# accuracy指标
def get_accuracy(outputs,labels,threshold=0.5):
    outputs = outputs > threshold
    labels = labels == torch.max(labels)
    corr = torch.sum(outputs==labels)
    tensor_size = outputs.size(0)*outputs.size(1)*outputs.size(2)*outputs.size(3)
    acc = float(corr)/float(tensor_size + 1e-6)
    # print(corr, tensor_size)
    return acc

# sensitivity指标
# Sensitivity == Recall
def get_sensitivity(outputs,labels,threshold=0.5):
    outputs = outputs > threshold
    labels = labels == torch.max(labels)
    # TP : True Positive
    # FN : False Negative
    TP = ((outputs==1).byte() + (labels==1).byte()) == 2
    FN = ((outputs==0).byte() + (labels==1).byte()) == 2
    # print(TP.size(), FN.size())
    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)     
    return SE

# precision指标
def get_precision(outputs,labels,threshold=0.5):    
    outputs = outputs > threshold
    labels = labels == torch.max(labels)

    # TP : True Positive
    # FP : False Positive
    TP = ((outputs==1).byte() + (labels==1).byte()) == 2
    FP = ((outputs==1).byte() + (labels==0).byte()) == 2
    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)
    # print("precision: ", (outputs==1) + (labels==0), " ", torch.sum(TP+FP), "\n")
    return PC
