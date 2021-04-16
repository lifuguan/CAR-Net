'''
Author: your name
Date: 2021-04-15 20:04:53
LastEditTime: 2021-04-15 22:20:56
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /leo/unetzoo/core/aceloss.py
'''
import time

import numpy as np
import torch
import torch.nn as nn

def ACELoss(y_pred, y_true, u=1, a=1, b=1):
    """
    Active Contour Loss
    based on total variations and mean curvature
    """
    def first_derivative(input):
        u = input
        m = u.shape[2]
        n = u.shape[3]

        ci_0 = (u[:, :, 1, :] - u[:, :, 0, :]).unsqueeze(2)
        ci_1 = u[:, :, 2:, :] - u[:, :, 0:m - 2, :]
        ci_2 = (u[:, :, -1, :] - u[:, :, m - 2, :]).unsqueeze(2)
        ci = torch.cat([ci_0, ci_1, ci_2], 2) / 2

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
        c_in = torch.ones_like(y_pred)
        c_out = torch.zeros_like(y_pred)
        region_in = torch.abs(torch.sum(y_pred * ((label - c_in) ** 2)))
        region_out = torch.abs(
            torch.sum((1 - y_pred) * ((label - c_out) ** 2)))
        region = u * region_in + region_out
        return region

    def elastica(input, a=1, b=1):
        ci, cj = first_derivative(input)
        cii, cjj, cij = second_derivative(input, ci, cj)
        beta = 1e-8
        length = torch.sqrt(beta + ci ** 2 + cj ** 2)
        curvature = (beta + ci ** 2) * cjj + \
                    (beta + cj ** 2) * cii - 2 * ci * cj * cij
        curvature = torch.abs(curvature) / ((ci ** 2 + cj ** 2) ** 1.5 + beta)
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