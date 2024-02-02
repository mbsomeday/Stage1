from sklearn.metrics import confusion_matrix
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib_venn
from PIL import Image
import os
from tqdm import tqdm
import argparse

from cv_models import DEVICE
from utils import dataset



# 根据混淆矩阵计算某类的召回率
# https://blog.csdn.net/a645676/article/details/127369713
def calculate_label_recall(confMatrix, model_name):
    l = len(confMatrix)
    for i in range(l):
        label_total_sum = confMatrix.sum(axis=0)[i]
        label_correct_sum = confMatrix[i][i]
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
        print(f'{model_name} recall of class {i}: {str(prediction)}%')


def temp_Veen():
    data_list = [{1, 2, 3}, {1, 2, 4}]
    m1_right_cases = {'a.pgm', 'b.pgm', 'c1.pgm'}
    m2_right_cases = {'a.pgm', 'b2.pgm', 'c.pgm'}
    m3_right_cases = {'a1.pgm', 'b.pgm', 'c1.pgm'}

    my_dpi = 150
    plt.figure(figsize=(600 / my_dpi, 600 / my_dpi), dpi=my_dpi)  # 控制图尺寸的同时，使图高分辨率（高清）显示
    g = matplotlib_venn.venn3(subsets=[m1_right_cases, m2_right_cases, m3_right_cases],  # 传入三组数据
                              set_labels=('Label 1', 'Label 2', 'Label 3'),  # 设置组名
                              set_colors=("#01a2d9", "#31A354", "#c72e29"),  # 设置圈的颜色，中间颜色不能修改
                              alpha=0.8,  # 透明度
                              normalize_to=1.0,  # venn图占据figure的比例，1.0为占满
                              )
    plt.show()


def image_filter(image_dir, txt_path):
    '''
        将EuroCity Pedestrian Dataset中，宽高比在[0.4, 0.6]之间，且原像素在
        的pedestrian记录下来

    Args:
        image_dir: EuroCity Pedestrian Dataset/Pedestrian
    '''
    # 规定条件
    ration_min = 0.4
    ration_max = 0.6

    image_for_test = []

    image_list = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
    for image_path in tqdm(image_list):
        image = Image.open(image_path)
        w, h = image.size
        resolution = w * h
        ratio = w / h
        if ratio > ration_min and ratio < ration_max and resolution <= 1500:
            image_for_test.append(image_path)
    # 将符合条件的image_path存储到txt文件中
    with open(txt_path, 'w') as f:
        for item in image_for_test:
            f.write(str(item) + '\n')
    return image_for_test


def get_risk(csv_path):
    '''
    此risk计算公式有待调整
    Args:
        csv_path:

    Returns:

    '''
    data = pd.read_csv(csv_path, header=None)
    FP = data.iloc[0, 1]
    FN = data.iloc[1, 0]

    sum1 = data.iloc[0, :].sum()
    sum2 = data.iloc[1, :].sum()
    total = sum1 + sum2

    risk = 1.0 * FN / total + 0.5 * FP / total

    return risk


# 制作segmentation数据集的txt
def get_dataseTxt(base_dir, txt_path):
    image_list = os.listdir(base_dir)
    with open(txt_path, 'a') as f:
        for image in image_list:
            image_path = os.path.join(base_dir, image)
            msg = image_path + ' ' + 'ped_examples 1\n'
            f.write(msg)



if __name__ == '__main__':
    base_dir = r'D:\chrom_download\DaimlerPedestrianDetectionBenchmark\PedCut2013_SegmentationDataset\data\completeData\left_images'
    txt_path = r'D:\my_phd\on_git\experiment\data\dataset_txt\DaiPedSegmentation\test.txt'
    get_dataseTxt(base_dir, txt_path)

