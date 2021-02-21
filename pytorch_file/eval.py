import torch
import torch.nn as nn
import torch.optim
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
    acc = float(corr)/float(tensor_size)
 
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

