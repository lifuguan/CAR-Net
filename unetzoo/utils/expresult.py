# -*- coding: utf-8 -*-

#
# Title: 实验结果保存类
# Descrip: 主要用于保存实验运行结果，包括日志，损失函数，度量值，模型并图形绘制等。
# Author: Cheng Yong
# Date: 2021-02-26
#

import os
import time
import matplotlib.pyplot as plt
import logging
import torch
from core.date import datehint
from utils.hyperparams import HyperParams


class ExpResult():
    def __init__(self, params):
        self.model = params.model
        self.loss = params.loss
        self.dataset = params.dataset
        self.epochs = params.epochs
        self.batch_size = params.batch_size

        self.expname = self.model + '-' + str(self.dataset) + "-" + str(self.epochs) + '-' + str(self.batch_size)
        self.exptime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 保存目录
        self.base_dir = params.savedir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        # 实验目录
        timepath = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self.exp_dir = os.path.join(self.base_dir, timepath + '-' + self.model + "-" + self.loss + "-" + self.dataset)
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        # 日志文件
        self.log_file = os.path.join(self.exp_dir,
                                     self.model + '-' + str(self.dataset) + '-' + str(self.epochs) + "-" + str(
                                         self.batch_size) + ".log")
        try:
            f = open(self.log_file, 'r')
            f.close()
        except IOError:
            f = open(self.log_file, 'w')

        # 训练目录
        self.train_dir = os.path.join(self.exp_dir, "train")
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        # 测试目录
        self.test_dir = os.path.join(self.exp_dir, "test")
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

        # 日志信息
        logging.basicConfig(
            filename=self.log_file,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )

        # ReadMe文件
        self.readme_file = os.path.join(self.exp_dir, "ReadMe.txt")
        # 控制台文件
        self.console_file = os.path.join(self.exp_dir, "Console.txt")
        # 损失文件
        self.loss_file = os.path.join(self.train_dir, "losses.txt")
        # 训练度量
        self.train_metrics_file = os.path.join(self.train_dir, "train_metrics.txt")
        # 测试度量
        self.test_metrics_file = os.path.join(self.test_dir, "test_metrics.txt")

    # 保存实验基本信息
    def expinfo(self):
        with open(self.readme_file, 'a') as f:
            f.write('=======' * 5 + '\n')
            f.write("实验名称：%s\n" % self.expname)
            f.write("实验时间：%s\n" % self.exptime)
            f.write("模型名称：%s\n" % self.model)
            f.write("数据集名称：%s\n" % self.dataset)
            f.write("epochs数目：%s\n" % self.epochs)
            f.write("batch_size大小：%s\n" % self.batch_size)
            f.write("实验目录：%s\n" % self.exp_dir)
            f.write('=======' * 5 + '\n')
            f.write('\n')

    # 保存控制台信息
    def print(self, str):
        with open(self.console_file, 'a') as f:
            f.write(str + '\n')

    # 保存日志信息
    def loginfo(self, str):
        logging.info(str)

    # 保存损失值
    def saveloss(self, descrip, loss):
        with open(self.loss_file, 'a') as f:
            f.write(datehint() + ': ' + descrip + '\n')
            f.write(str(loss) + '\n')
            f.write('\n')

    # 保存损失值列表
    def savelosses(self, descrip, losses):
        with open(self.loss_file, 'a') as f:
            f.write(datehint() + ": " + descrip + '\n')
            s = '['
            for i in range(len(losses) - 1):
                s += str(losses[i]) + " ,"
            s += str(losses[len(losses) - 1]) + ']\n'  # 去除单引号，逗号，每行末尾追加换行符
            f.write(s)
            f.write('\n')

    # 绘制损失函数曲线
    def plotloss(self, descrip, losses):
        num = len(loss)
        x = [i for i in range(num)]
        pic_file = os.path.join(self.train_dir, descrip + '-' + str(time.time()) + '-loss.jpg')
        plt.figure()
        plt.plot(x, losses, label='Loss')
        plt.legend()
        plt.savefig(pic_file)

    # 保存模型性能度量值
    def savemetric(self, train, descrip, metric):
        if train == 'train':
            file = self.train_metrics_file
        else:
            file = self.test_metrics_file
        with open(file, 'a') as f:
            f.write(datehint() + ": " + descrip + '\n')
            f.write(str(metric) + '\n')
            f.write('\n')

    # 保存模型性能度量列表
    def savemetrics(self, train, descrips, *args):
        if train == 'train':
            file = self.train_metrics_file
        else:
            file = self.test_metrics_file
        with open(file, 'a') as f:
            f.write(datehint() + ": " + descrips + '\n')
            name = descrips.split('-')
            metrics_value = args
            n = 0
            for v in metrics_value:
                f.write(name[n] + '\n')
                s = '['
                for i in range(len(v) - 1):
                    s += str(v[i]) + ", "
                s += str(v[len(v) - 1]) + ']\n'  # 去除单引号，逗号，每行末尾追加换行符
                f.write(s)
                n = n + 1
            f.write('\n\n')

    # 绘制模型性能度量曲线
    def plotmetrics(self, train, descrips, *args):
        names = descrips.split('-')
        metrics_value = args
        num = len(metrics_value[0])
        i = 0
        x = [i for i in range(num)]
        if train == 'train':
            pic_file = os.path.join(self.train_dir, descrips + '-' + str(time.time()) + '-metrics.jpg')
        else:
            pic_file = os.path.join(self.test_dir, descrips + '-' + str(time.time()) + '-metrics.jpg')
        plt.figure()
        for m in metrics_value:
            plt.plot(x, m, label=str(names[i]))
            # plt.scatter(x,l,label=str(l))
            i += 1
        plt.legend()
        plt.savefig(pic_file)

    # 保存模型
    def savemodel(self, model):
        '''
        :param model: model
        :return:
        '''
        path = os.path.join(self.train_dir, str(self.model) + '-' + str(self.dataset) + '-' + str(self.epochs)
                            + '-' + str(self.batch_size) + '.pth')
        torch.save(model.state_dict(), path)

    # 载入模型
    def loadmodel(self, model):
        '''
        :param model: model
        :return:
        '''
        path = os.path.join(self.train_dir, str(self.model) + '-' + str(self.dataset) + '-' + str(self.epochs)
                            + '-' + str(self.batch_size) + '.pth')
        model.load_state_dict(torch.load(path))
        return model


"""main"""
if __name__ == '__main__':
    params = HyperParams("E:/studio/learn/python/src/lab/unetzoo/config/unet.json")
    result = ExpResult(params)
    result.expinfo()
    result.print("========start========")
    for i in range(20):
        result.print("%i / 200" % (i))
        result.loginfo("%i / 200" % (i))
        # 损失值
        result.saveloss('3-6', 0.6)
        loss = [0.345, 0.323, 0.444, 0.333, 0.321, 0.245, 0.234, 0.222, 0.111]
        result.savelosses('1-5', loss)
        loss = [0.345, 0.323, 0.444, 0.333, 0.321, 0.245, 0.234, 0.222, 0.111]
        result.savelosses('2-5', loss)
        loss = [0.345, 0.323, 0.444, 0.333, 0.321, 0.245, 0.234, 0.222, 0.111]
        result.savelosses('3-5', loss)
        result.plotloss('3-5', loss)

        # 性能度量值(一种度量指标，长度要求相等)
        metrics1 = [0.345, 0.323, 0.444, 0.333, 0.321, 0.245, 0.234, 0.222, 0.111]
        result.savemetrics('train', 'metrics1', metrics1)
        result.plotmetrics('train', 'metrics1', metrics1)

        # 性能度量值(三种度量指标，长度要求相等)
        metrics1 = [0.345, 0.323, 0.444, 0.333, 0.321, 0.245, 0.234, 0.222, 0.111]
        metrics2 = [1.345, 1.323, 2.444, 2.333, 3.321, 3.245, 4.234, 3.222, 2.111]
        metrics3 = [3.345, 2.323, 6.444, 7.333, 5.321, 8.245, 9.234, 10.222, 11.111]
        result.savemetric('train', 'metrics1', 0.3)
        result.savemetrics('test', 'metrics1-metrics2-metrics3', metrics1, metrics2, metrics3)
        result.plotmetrics('test', 'metrics1-metrics2-metrics3', metrics1, metrics2, metrics3)

        # 性能度量值(一种度量指标，长度要求相等)
        metrics1 = [0.345, 0.323, 0.444, 0.333, 0.321, 0.245, 0.234, 0.222, 0.111]
        result.savemetrics('train', 'metrics1', metrics1)
        result.plotmetrics('train', 'metrics1', metrics1)

        # 性能度量值(三种度量指标，长度要求相等)
        metrics1 = [0.345, 0.323, 0.444, 0.333, 0.321, 0.245, 0.234, 0.222, 0.111]
        metrics2 = [1.345, 1.323, 2.444, 2.333, 3.321, 3.245, 4.234, 3.222, 2.111]
        metrics3 = [3.345, 2.323, 6.444, 7.333, 5.321, 8.245, 9.234, 10.222, 11.111]
        result.savemetrics('test', 'metrics1-metrics2-metrics3', metrics1, metrics2, metrics3)
        result.plotmetrics('test', 'metrics1-metrics2-metrics3', metrics1, metrics2, metrics3)
