# 定义 Dataset
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
from torchvision import transforms

from cv_models import DEVICE


def get_image_transform(mode):
    '''
    Args:
        mode: 0, 1, 2 -> return single transformer
                -1 -> return transformer list
    '''
    image_transform = [
        transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 彩色图像转灰度图像num_output_channels默认1
            transforms.Resize([36, 18]),  # [h, w])
            transforms.ToTensor()
        ]),
        transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 彩色图像转灰度图像num_output_channels默认1
            transforms.Resize([36, 18]),  # [h, w])

            transforms.RandomHorizontalFlip(p=1),  # 水平翻转
            transforms.ToTensor()
        ]),
        transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 彩色图像转灰度图像num_output_channels默认1
            transforms.Resize([36, 18]),  # [h, w])

            transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
            transforms.ToTensor()
        ])
    ]
    if mode >= 0:
        return image_transform[mode]
    else:
        return image_transform


class MyDataset(Dataset):
    def __init__(self, running_on, txt_name, transformer_mode=None, multinput=False):

        super(MyDataset).__init__()
        self.image_dir = running_on['vars']['base_dir']
        self.txt_dir = running_on['vars']['txt_dir']
        self.txt_name = txt_name
        self.multinput = multinput
        self.transformer_mode = -1 if multinput else transformer_mode
        self.image_transformer = get_image_transform(self.transformer_mode)

        txt_path = os.path.join(self.txt_dir, txt_name)
        with open(txt_path, 'r') as f:
            data = f.readlines()

        images = []
        labels = []
        for line in data:
            line = line.rstrip()
            word = line.split()
            images.append(os.path.join(self.image_dir, word[0]))
            labels.append(word[2])

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_name = self.images[item]
        label = self.labels[item]
        label = np.array(label).astype(np.int64)
        img = Image.open(image_name)  # PIL image shape:（C, W, H）

        if not self.multinput:
            # 单输入的情况
            img = self.image_transformer(img)
            return img, label, image_name
        else:
            # 需要对图片进行多种变化的情况
            image_list = []
            for trans in self.image_transformer:
                image_list.append(trans(img))
            return image_list, label, image_name


class Dataset_for_ECPD(Dataset):
    def __init__(self, image_dir, txt_path, transform=None):
        self.root = image_dir
        self.txt_path = txt_path
        with open(self.txt_path, 'r') as f:
            data = f.readlines()
        images = []
        labels = []
        for line in data:
            line = line.rstrip()
            images.append(line)
            labels.append('1')
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img = self.images[item]
        label = self.labels[item]
        img = Image.open(img)  # PIL image shape:（C, W, H）
        if self.transform is not None:
            img = self.transform(img)
        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)
        return img, label


class dataset_for_Daimler(Dataset):
    def __init__(self, image_dir, transform):
        super(dataset_for_Daimler).__init__()
        self.image_dir = image_dir
        self.images = [os.path.join(self.image_dir, image) for image in os.listdir(self.image_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.images[item]
        label = 1
        img = Image.open(img)  # PIL image shape:（C, W, H）
        if self.transform is not None:
            img = self.transform(img)

        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)
        return img, label
























