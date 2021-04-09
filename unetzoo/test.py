'''
Author: your name
Date: 2021-04-09 18:18:40
LastEditTime: 2021-04-09 20:56:12
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /leo/unetzoo/test.py
'''
import argparse
import os
from torch import optim
from torchvision import transforms
import torch
from core import getDataset
from model import getModel
from utils.expresult import ExpResult
from utils.hyperparams import HyperParams
from andytrainer import train
from andytrainer import test