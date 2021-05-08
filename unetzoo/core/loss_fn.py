'''
Author: your name
Date: 2021-04-15 20:04:53
LastEditTime: 2021-05-06 23:50:18
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /leo/unetzoo/core/aceloss.py
'''
import time

import numpy as np
import torch
import torch.nn as nn
class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()
 
	def	forward(self, input, target):
		N = target.size(0)
		smooth = 1
 
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
 
		intersection = input_flat * target_flat
 
		loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1 - loss.sum() / N
 
		return loss

def ACELoss(y_pred, y_true, u=1, a=1, b=1):
    """
    Active Contour Loss
    based on total variations and mean curvature
    """
    def first_derivative(input):
        """
        一阶微分
        """
        u = input
        m = u.shape[2]  # 获取高
        n = u.shape[3]  # 获取宽

        ci_0 = (u[:, :, 1, :] - u[:, :, 0, :]).unsqueeze(2) # Height方向梯度，且i=0
        ci_1 = u[:, :, 2:, :] - u[:, :, 0:m - 2, :]      # Height方向梯度，且i=2:m-1
        ci_2 = (u[:, :, -1, :] - u[:, :, m - 2, :]).unsqueeze(2) # Height方向梯度，且i=m
        ci = torch.cat([ci_0, ci_1, ci_2], 2) / 2   # 在Height处concat

        # 在width方向求梯度
        cj_0 = (u[:, :, :, 1] - u[:, :, :, 0]).unsqueeze(3)
        cj_1 = u[:, :, :, 2:] - u[:, :, :, 0:n - 2]
        cj_2 = (u[:, :, :, -1] - u[:, :, :, n - 2]).unsqueeze(3)
        cj = torch.cat([cj_0, cj_1, cj_2], 3) / 2

        return ci, cj

    def second_derivative(input, ci, cj):
        u = input
        m = u.shape[2]
        n = u.shape[3]

        cii_0 = (u[:, :, 1, :] + u[:, :, 0, :] -
                 2 * u[:, :, 0, :]).unsqueeze(2)
        cii_1 = u[:, :, 2:, :] + u[:, :, :-2, :] - 2 * u[:, :, 1:-1, :]
        cii_2 = (u[:, :, -1, :] + u[:, :, -2, :] -
                 2 * u[:, :, -1, :]).unsqueeze(2)
        cii = torch.cat([cii_0, cii_1, cii_2], 2)

        cjj_0 = (u[:, :, :, 1] + u[:, :, :, 0] -
                 2 * u[:, :, :, 0]).unsqueeze(3)
        cjj_1 = u[:, :, :, 2:] + u[:, :, :, :-2] - 2 * u[:, :, :, 1:-1]
        cjj_2 = (u[:, :, :, -1] + u[:, :, :, -2] -
                 2 * u[:, :, :, -1]).unsqueeze(3)

        cjj = torch.cat([cjj_0, cjj_1, cjj_2], 3)

        cij_0 = ci[:, :, :, 1:n]
        cij_1 = ci[:, :, :, -1].unsqueeze(3)

        cij_a = torch.cat([cij_0, cij_1], 3)
        cij_2 = ci[:, :, :, 0].unsqueeze(3)
        cij_3 = ci[:, :, :, 0:n - 1]
        cij_b = torch.cat([cij_2, cij_3], 3)
        cij = cij_a - cij_b

        return cii, cjj, cij

    def region(y_pred, y_true, u=1):
        label = y_true.float()
        c_in = torch.ones_like(y_pred)   # 生成大小和y_pred大小相同，全为1的张量
        c_out = torch.zeros_like(y_pred) # 生成大小和y_pred大小相同，全为0的张量
        region_in = torch.abs(torch.sum(y_pred * ((label - c_in) ** 2)))
        region_out = torch.abs(
            torch.sum((1 - y_pred) * ((label - c_out) ** 2)))
        region = u * region_in + region_out
        return region

    def elastica(input, a=0.001, b=2):
        ci, cj = first_derivative(input)  # 一阶偏导
        cii, cjj, cij = second_derivative(input, ci, cj) # 二阶偏导
        beta = 1e-8
        length = torch.sqrt(beta + ci ** 2 + cj ** 2)
        curvature = (beta + 1 + ci ** 2) * cjj + \
                    (beta + 1 + cj ** 2) * cii - 2 * ci * cj * cij     # 与论文不一样，缺少 +1
        curvature = torch.abs(curvature) / ((ci ** 2 + cj ** 2) ** 1.5 + 1 + beta) # 与论文不一样，缺少 +1
        elastica = torch.sum((a + b * (curvature ** 2)) * torch.abs(length))
        return elastica

    loss = region(y_pred, y_true, u=u) + elastica(y_pred, a=a, b=b)
    return loss

if __name__ == "__main__":
    "test demo"
    x2 = torch.rand((2, 3, 97, 80))
    y2 = torch.rand((2, 3, 97, 80))
    time3 = time.time()
    print("ACELoss:", ACELoss(x2, y2).item())
    print(time.time() - time3)