from torchvision import transforms
import argparse
import torch
from torch.utils.data import DataLoader
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm

from diversity import ensemble_models
from cv_models import basic_learners, DEVICE
from utils.dataset import MyDataset
from utils import dataset


def test_model(test_dataset, test_loader, model, model_name, is_ensemble=False, ensemble_type=None):
    '''
        model_name: name of one model or 'SoftVoting' / 'HardVoting'
    '''

    num_correct = 0
    hard_examples = []

    # 预测结果和真实标签
    y_pred = []
    y_true = []

    with torch.no_grad():
        for index, data in enumerate(tqdm(test_loader)):
            images, labels, image_names = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            if is_ensemble:
                if ensemble_type == 'soft':
                    out = ensemble_models.single_input_soft_voting(model_list=model, images=images)
                    _, pred = torch.max(out, 1)
                else:
                    out = ensemble_models.single_input_hard_voting(model_list=model, images=images)
                    pred = out
            else:
                model.eval()
                out = model(images)
                _, pred = torch.max(out, 1)

            # 将label和pred加入列表中
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

            num_correct += (pred == labels).sum()

            # 为混淆矩阵做准备
            all_index = torch.arange(start=0, end=(images.shape[0])).to(DEVICE)
            wrong_idx = all_index[pred != labels]
            for w_idx in wrong_idx:
                if ensemble_type == 'soft':
                    wrongCase_info = image_names[w_idx] + ' ' + str(out[w_idx, :].tolist()) + ' ' + str(labels[w_idx].item())
                else:
                    wrongCase_info = image_names[w_idx] + ' ' + str(out[w_idx]) + ' ' + str(labels[w_idx].item())
                hard_examples.append(wrongCase_info)

    # 写入混淆矩阵
    write_to_dir = r'D:\my_phd\on_git\experiment\data'
    cm_name = f'{model_name}.csv'
    cm_path = os.path.join(write_to_dir, 'Confusion_metrics', cm_name)
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm).to_csv(cm_path, index=False, header=False)
    print(f'Successfully Confusion metric of {model_name} file written to {cm_path}!')

    # 写入错分样本
    hard_example_name = f'{model_name}.txt'
    hard_example_path = os.path.join(write_to_dir, 'Hard_example_predictions', hard_example_name)
    with open(hard_example_path, 'w') as f:
        for item in hard_examples:
            f.write(item + '\n')
    print('Hard examples written successfully!')
    print('错分样本数量:', len(hard_examples))

    # 输出正确率
    test_accuracy = num_correct / len(test_dataset)
    print(f'{model_name} accuracy is "{test_accuracy:.10f}')


def multipleInput_voting(test_dataset, test_loader, model_list, ensemble_type='hard'):
    model_name = 'multipleInput_softVoting' if ensemble_type=='soft' else 'multipleInput_hardVoting'
    num_correct = 0
    hard_examples = []

    # 预测结果和真实标签
    y_pred = []
    y_true = []

    with torch.no_grad():
        for index, data in enumerate(test_loader):
            # print(index)
            images, labels, image_names = data

            image_num = images[0].shape[0]

            img0 = images[0].to(DEVICE)
            img1 = images[1].to(DEVICE)
            img2 = images[2].to(DEVICE)

            labels = labels.to(DEVICE)

            out1 = model_list[0](img0)
            out2 = model_list[1](img1)
            out3 = model_list[2](img2)

            if ensemble_type == 'soft':
                out = ensemble_models.multiple_input_soft_voting([out1, out2, out3])
                _, pred = torch.max(out, 1)
                pred = pred.to(DEVICE)
            else:
                out = ensemble_models.multiple_input_hard_voting([out1, out2, out3])
                pred = out

            # 将label和pred加入列表中
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

            num_correct += (pred == labels).sum()

            # 为混淆矩阵做准备
            all_index = torch.arange(start=0, end=image_num).to(DEVICE)
            wrong_idx = all_index[pred != labels]
            for w_idx in wrong_idx:
                if ensemble_type == 'soft':
                    wrongCase_info = image_names[w_idx] + ' ' + str(out[w_idx, :].tolist()) + ' ' + str(labels[w_idx].item())
                else:
                    wrongCase_info = image_names[w_idx] + ' ' + str(out[w_idx]) + ' ' + str(labels[w_idx].item())
                hard_examples.append(wrongCase_info)

    # 写入混淆矩阵
    write_to_dir = r'D:\my_phd\on_git\experiment\data'
    cm_name = f'{model_name}.csv'
    cm_path = os.path.join(write_to_dir, 'Confusion_metrics', cm_name)
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm).to_csv(cm_path, index=False, header=False)
    print(f'Successfully Confusion metric of {model_name} file written to {cm_path}!')

    # 写入错分样本
    hard_example_name = f'{model_name}.txt'
    hard_example_path = os.path.join(write_to_dir, 'Hard_example_predictions', hard_example_name)
    with open(hard_example_path, 'w') as f:
        for item in hard_examples:
            f.write(item + '\n')
    print(f'Wrong examples written successfully!')
    print('错分样本数量:', len(hard_examples))

    # 输出正确率
    test_accuracy = num_correct / len(test_dataset)
    print(f'{model_name} accuracy is "{test_accuracy:.10f}')



