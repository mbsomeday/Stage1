from sklearn.metrics import confusion_matrix
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib_venn
from PIL import Image
import os
from tqdm import tqdm
import argparse
import re

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


def re_write_txt_file(org_txt_pat, dest_txt_path):
    '''
        将pred txt文件增添新列内容
    '''

    with open(org_txt_pat, 'r') as f:
        data = f.readlines()

    conf_cls_data = []

    for idx, item in enumerate(data):
        item = item.strip()

        right_label = int(item[-1])
        wrong_label = 1 - right_label
        processed = re.sub(r'[\[\]]', '*', item)

        processed = processed.split('*')
        preds = processed[1].split(',')

        conf_on_wrong_cls = float(preds[wrong_label])

        if conf_on_wrong_cls < 0.7:
            conf_cls = 'low'
        elif conf_on_wrong_cls < 0.9:
            conf_cls = 'medium'
        else:
            conf_cls = 'high'

        msg = item + ' ' + conf_cls
        conf_cls_data.append(msg)

    with open(dest_txt_path, 'w') as f:
        for item in conf_cls_data:
            f.write(item + '\n')


def get_overConfExamples(txt_path):
    with open(txt_path, 'r') as f:
        data = f.readlines()
    over_conf_examples = []
    for item in data:
        item = item.strip()
        content = item.split(' ')
        conf_cls = content[-1]
        if conf_cls == 'high':
            over_conf_examples.append(content[0])

    return over_conf_examples


def get_saved_num(a, b, c):
    inter_abc = list(set(a).intersection(b, c))
    inter_ab = list(set(a).intersection(b))
    inter_ac = list(set(a).intersection(c))
    inter_bc = list(set(b).intersection(c))

    union_abc = list(set(a).union(b, c))  # 求多个list的并集

    save_num = len(union_abc) - len(inter_ab) - len(inter_ac) - len(inter_bc) + 2 * len(inter_abc)

    print('len(union_abc):', len(union_abc))
    print('saved_num:', save_num)


if __name__ == '__main__':
    org_path = r'D:\my_phd\on_git\experiment\data\Model_Evaluation\Hardexample_Predictions\DaiPedClassify\Baseline\CNN.txt'
    dest_path = r'D:\my_phd\on_git\experiment\data\Model_Evaluation\Hardexample_Predictions\DaiPedClassify\Baseline\test.txt'
    re_write_txt_file(org_txt_pat=org_path, dest_txt_path=dest_path)















