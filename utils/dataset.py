# 定义 Dataset
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
from torchvision import transforms


class Fake_dataset(Dataset):
    def __init__(self, total_num=500):
        super(Fake_dataset).__init__()
        self.images = torch.rand(size=(total_num, 1, 36, 18))
        self.labels = []
        for i in range(total_num):
            self.labels.append(1)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        label = np.array(label).astype(np.int64)
        return img, label


class MyDataset(Dataset):
    def __init__(self, base_dir, txt_dir=None, txt_name=None, transform=None):
        super(MyDataset, self).__init__()
        self.root = base_dir
        self.txt_dir = txt_dir
        txt_path = os.path.join(self.txt_dir, txt_name)
        with open(txt_path, 'r') as f:
            data = f.readlines()
        images = []
        labels = []
        for line in data:
            line = line.rstrip()
            word = line.split()
            images.append(os.path.join(self.root, word[0]))
            labels.append(word[2])

        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image_name = self.images[item]
        label = self.labels[item]
        img = Image.open(image_name)  # PIL image shape:（C, W, H）
        if self.transform is not None:
            img = self.transform(img)
        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)
        return img, label, image_name


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


def get_image_transform(mode):
    if mode == 1:
        image_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    elif mode == 2:
        image_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),  # 水平翻转
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.ToTensor()
        ])
    else:
        image_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
            transforms.ToTensor()
        ])
    return image_transform


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


def get_dataloader(image_dir, txt_dir, txt_name, transformer_mode):
    img_transformer = get_image_transform(transformer_mode)
    ret_dataset = MyDataset(base_dir=image_dir, txt_dir=txt_dir, txt_name=txt_name, transform=img_transformer)
    ret_loader = DataLoader(dataset=ret_dataset, batch_size=64, shuffle=False, drop_last=False)

    return ret_dataset, ret_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--txt_dir', type=str, required=True, help='dir path that save image split .txt')
    parser.add_argument('--type', type=str, choices=['train', 'test', 'val'], required=True)
