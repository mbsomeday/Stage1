# pedestrian classification dataset
# https://blog.csdn.net/qq_53345829/article/details/124308515

# import torch
import os
import shutil
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


np.random.seed(13)

BASE_DIR = r'D:\chrom_download\DaimlerPedestrianDetectionBenchmark\DC-ped-dataset_base'
PED_DIR = r'D:\chrom_download\DaimlerPedestrianDetectionBenchmark\DC-ped-dataset_base\pedestrian'
NON_PED_DIR = r'D:\chrom_download\DaimlerPedestrianDetectionBenchmark\DC-ped-dataset_base\nonPedestrian'

TRAIN_TXT = r'D:\chrom_download\DaimlerPedestrianDetectionBenchmark\DC-ped-dataset_base\train.txt'
TEST_TXT = r'D:\chrom_download\DaimlerPedestrianDetectionBenchmark\DC-ped-dataset_base\test.txt'
VAL_TXT = r'D:\chrom_download\DaimlerPedestrianDetectionBenchmark\DC-ped-dataset_base\val.txt'

def rename(dir_path):
    files = os.listdir(dir_path)

    for idx, f in enumerate(files):
        f_path = os.path.join(dir_path, f)
        new_name = str(idx).zfill(5) + '.pgm'
        new_path = os.path.join(dir_path, new_name)
        os.rename(f_path, new_path)


def write_to_txt(txt_path, image_list, cls_name, cls_code):
    with open(txt_path, 'a+') as f:
        for img in image_list:
            image_path = os.path.join(cls_name, img)
            # 格式：class/image_name + label_name + label_code
            line = image_path + ' ' + str(cls_name) + ' ' + str(cls_code) + '\n'
            f.write(line)


def split_single_cls(base_dir, cls_name, cls_code):
    train_ratio = 0.6
    val_traio = 0.2
    test_ratio = 0.2
    images = os.listdir(os.path.join(base_dir, cls_name))
    image_num = len(images)

    # 将序号打乱
    shuffled_images = np.random.permutation(images)
    train_size = int(image_num * train_ratio)
    test_size = int(image_num * test_ratio)
    val_size = int(image_num * val_traio)

    train_images = shuffled_images[: train_size]
    test_images = shuffled_images[train_size: train_size + test_size]
    val_images = shuffled_images[train_size + test_size:]

    # 写入.txt文件中
    train_path = os.path.join(base_dir, 'train.txt')
    test_path = os.path.join(base_dir, '../test.txt')
    val_path = os.path.join(base_dir, 'val.txt')

    write_to_txt(txt_path=train_path, image_list=train_images, cls_name=cls_name, cls_code=cls_code)
    write_to_txt(txt_path=test_path, image_list=test_images, cls_name=cls_name, cls_code=cls_code)
    write_to_txt(txt_path=val_path, image_list=val_images, cls_name=cls_name, cls_code=cls_code)


def dataset_split(base_dir):
    '''
    base_dir文件结构：所有不同类别有各自的文件夹，文件夹内有图片
    Args:
        base_dir:
        class_dir:
    Returns: 将base_dir下的文件根据0.6，0.2，0.2划分为train,test,val
            按照 image_path label_name lanel_code格式写入
    '''

    # 1. 获取所有类别
    file_list = os.listdir(base_dir)
    cls_list = []
    for file in file_list:
        file_path = os.path.join(base_dir, file)
        if os.path.isdir(file_path):
            cls_list.append(file)
    cls_num = len(cls_list)

    # 2. 遍历每个类，划分train, test, val
    for idx, cls in enumerate(cls_list):
        split_single_cls(base_dir, cls, idx)


# dataset_split(BASE_DIR)


transforms = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, base_dir, txt_name=None, transform=None):
        super(MyDataset, self).__init__()
        self.root = base_dir
        txt_path = os.path.join(self.root, txt_name)
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
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img)
        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)
        return img, label


train_dataset = MyDataset(base_dir=BASE_DIR, txt_name='train.txt', transform=transforms)
train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

val_dataset = MyDataset(base_dir=BASE_DIR, txt_name='val.txt', transform=transforms)
val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=True)


# for i, data in enumerate(train_loader):
#     images, labels = data
#     print(labels)
#     break



































