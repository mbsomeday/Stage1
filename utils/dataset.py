# 定义 Dataset
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os


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
        img = self.images[item]
        label = self.labels[item]
        img = Image.open(img)  # PIL image shape:（C, W, H）
        if self.transform is not None:
            img = self.transform(img)
        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)
        return img, label
