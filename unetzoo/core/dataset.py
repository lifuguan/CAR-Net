# -*- coding: utf-8 -*-
#
# Title: UNET-ZOO Dataset
# URL: https://github.com/Andy-zhujunwen/UNET-ZOO
# Date: 2020-10-23
#

import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import numpy as np
import cv2
from glob import glob
import imageio


class DriveEyeDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r''
        self.train_root = r"dataset/drive/training"
        self.val_root = r"dataset/drive/test"
        self.test_root = self.val_root
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        pics = []
        masks = []
        assert self.state =='train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            root = self.train_root
            n = len(os.listdir(root+"/images/"))
            for i in range(n):
                img = os.path.join(root+"/images/", "%d_training.tif" % (i+21)) 
                mask = os.path.join(root+"/1st_manual/", "%d_manual1.gif" % (i+21))
                pics.append(img)
                masks.append(mask)
        if self.state == 'val':
            root = self.val_root
            n = len(os.listdir(root+"/images/"))
            for i in range(n):
                img = os.path.join(root+"/images/", "%02d_test.tif" % (i+1)) 
                mask = os.path.join(root+"/1st_manual/", "%02d_manual1.gif" % (i+1))
                pics.append(img)
                masks.append(mask)
        if self.state == 'test':
            root = self.test_root
            n = len(os.listdir(root+"/images/"))
            for i in range(n):
                img = os.path.join(root+"/images/", "%02d_test.tif" % (i+1)) 
                mask = os.path.join(root+"/1st_manual/", "%02d_manual1.gif" % (i+1))
                pics.append(img)
                masks.append(mask)
        
        return pics,masks

    def __getitem__(self, index):
        imgx,imgy=(576,576)
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        #print(pic_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        if mask == None:
            mask = imageio.mimread(mask_path)
            mask = np.array(mask)[0]
        pic = cv2.resize(pic,(imgx,imgy))
        mask = cv2.resize(mask, (imgx, imgy))
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255

        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)



class DSB2018CellDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'dataset/dsb2018'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths = None,None
        self.train_mask_paths, self.val_mask_paths = None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + '/images/*')
        self.mask_paths = glob(self.root + '/masks/*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.val_img_paths,self.val_mask_paths  #因数据集没有测试集，所以用验证集代替

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)



class EsophagusDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.train_root = r"dataset/data_sta_all/train_data"
        self.val_root = r"dataset/data_sta_all/test_data"
        self.test_root = self.val_root
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root
        pics = []
        masks = []
        n = len(os.listdir(root)) // 2  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        for i in range(n):
            img = os.path.join(root, "%05d.png" % i)  # liver is %03d
            mask = os.path.join(root, "%05d_mask.png" % i)
            pics.append(img)
            masks.append(mask)
            #imgs.append((img, mask))
        return pics,masks

    def __getitem__(self, index):
        #x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        origin_x = cv2.imread(x_path)
        origin_y = cv2.imread(y_path,cv2.COLOR_BGR2GRAY)
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y,x_path,y_path

    def __len__(self):
        return len(self.pics)



class ISBICellDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'dataset/isbi'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'/train/images/*')
        self.mask_paths = glob(self.root + r'/train/label/*')
        # self.val_img_paths = glob(self.root + r'/val/val_images/*')
        # self.val_mask_paths = glob(self.root + r'/val/val_mask/*')
        # self.test_img_paths = glob(self.root + r'/test/test_images/*')
        # self.test_mask_paths = glob(self.root + r'/test/test_mask/*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths,self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths,self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)




class KaggleLungDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'dataset/finding-lungs-in-ct-data'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'/2d_images/*')
        self.mask_paths = glob(self.root + r'/2d_masks/*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths, self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class LiverDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'dataset/liver'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'/img/*')
        self.mask_paths = glob(self.root + r'/mask/*')

        # sklearn函数：分离训练和验证&测试集
        self.train_img_paths, val_test_img_paths, self.train_mask_paths, val_test_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)

        self.test_img_paths, self.val_img_paths, self.test_mask_paths,self.val_mask_paths = \
            train_test_split(val_test_img_paths, val_test_mask_paths, test_size=0.5, random_state=41)
        self.test_img_paths.clear()
        self.test_mask_paths.clear()
        self.test_img_paths.append('dataset/liver/img/400.png')
        self.test_mask_paths.append('dataset/liver/mask/400.png')
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths,self.test_mask_paths
        


    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        pic = cv2.imread(pic_path)
        pic = cv2.resize(pic, (576, 576))
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (576, 576))
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)


class CornealDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'dataset/corn'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.train_img_paths = glob(self.root + r'/training/train_images/*')
        self.train_mask_paths = glob(self.root + r'/training/train_mask/*')
        self.val_img_paths = glob(self.root + r'/val/val_images/*')
        self.val_mask_paths = glob(self.root + r'/val/val_mask/*')
        self.test_img_paths = glob(self.root + r'/test/test_images/*')
        self.test_mask_paths = glob(self.root + r'/test/test_mask/*')
        # self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
        #     train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths,self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)


class RaceCarDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.train_root = r"dataset/RaceCar/train"
        self.val_root = r"dataset/RaceCar/val"
        self.test_root = self.val_root
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        n = len(os.listdir(root)) // 3  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        for i in range(n):
            img = os.path.join(root, "%04d.jpg" % (i+10))  # liver is %03d
            mask = os.path.join(root, "%04d_mask.jpg" % (i+10))
            pics.append(img)
            masks.append(mask)
            #imgs.append((img, mask))
        return pics,masks

    def __getitem__(self, index):
        imgx,imgy=(576,576)
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _,mask = cv2.threshold(mask, 30,255,cv2.THRESH_BINARY)
        # if mask == None:
        #     mask = imageio.mimread(mask_path)
        #     mask = np.array(mask)[0]
        pic = cv2.resize(pic,(imgx,imgy))
        mask = cv2.resize(mask, (imgx, imgy))
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)



class Covid19Dataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'dataset/covid19'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'/img/*')
        self.mask_paths = glob(self.root + r'/mask/*')

        # sklearn函数：分离训练和验证集
        self.train_img_paths, val_test_img_paths, self.train_mask_paths, val_test_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)

        self.test_img_paths, self.val_img_paths, self.test_mask_paths,self.val_mask_paths = \
            train_test_split(val_test_img_paths, val_test_mask_paths, test_size=0.5, random_state=41)
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths,self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        pic = cv2.imread(pic_path)
        pic = cv2.resize(pic, (576, 576))
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (576, 576))
        mask = np.expand_dims(mask, axis=2)
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)


class LungDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'dataset/lung'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'/img/*')
        self.mask_paths = glob(self.root + r'/mask/*')

        # sklearn函数：分离训练和验证集
        self.train_img_paths, val_test_img_paths, self.train_mask_paths, val_test_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)

        self.test_img_paths, self.val_img_paths, self.test_mask_paths,self.val_mask_paths = \
            train_test_split(val_test_img_paths, val_test_mask_paths, test_size=0.5, random_state=41)
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths,self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        pic = cv2.imread(pic_path)
        pic = cv2.resize(pic, (576, 576))
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (576, 576))
        mask = np.expand_dims(mask, axis=2)
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)
