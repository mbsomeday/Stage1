from torchvision import transforms
import argparse
import torch
from torch.utils.data import DataLoader
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

from cv_models import DEVICE, diversity


def write_CM_HardExamples(runningOn, isCM, isHardExample, **kwargs):
    write_to_dir = runningOn['vars']['evaluation_dir']

    model_name = kwargs["model_name"]
    dataset_name = kwargs["dataset_name"]
    dir_name = kwargs['dir_name']
    file_suffix = '' if dir_name == 'Baseline' else '_' + kwargs['ensemble_type']

    # 写入混淆矩阵
    if isCM:
        cm_name = f'{model_name}{file_suffix}.csv'
        cm_path = os.path.join(write_to_dir, 'Confusion_Metrics', kwargs["dataset_name"], dir_name, cm_name)
        cm = confusion_matrix(kwargs['y_true'], kwargs['y_pred'])
        pd.DataFrame(cm).to_csv(cm_path, index=False, header=False)
        print('CM:\n', cm)
        print(f'Successfully Confusion metric of {model_name} file written to {cm_path}!')

    # 写入错分样本
    if isHardExample:
        hard_examples = kwargs["hard_examples"]

        hard_example_name = f'{model_name}{file_suffix}.txt'
        hard_example_path = os.path.join(write_to_dir, 'Hardexample_Predictions', dataset_name, dir_name,
                                         hard_example_name)
        with open(hard_example_path, 'w') as f:
            for item in hard_examples:
                f.write(item + '\n')
        print(f'Hard example predictions of {model_name} file written to {hard_example_path} successfully!')
        print('错分样本数量:', len(hard_examples))


def test_singleInput(runningOn, test_dataset, test_loader, model=None, model_list=None, **kwargs):
    model_name = kwargs["model_name"]
    dataset_name = kwargs["dataset_name"]
    ensemble_type = kwargs["ensemble_type"]

    print('-' * 30 + 'Start Testing' + '-' * 30)

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
            if ensemble_type == 'soft':
                out = diversity.softVoting(inpuType='Single', model_list=model_list, images=images)
                _, pred = torch.max(out, 1)
                pred = pred.to(DEVICE)
            elif ensemble_type == 'hard':
                out = diversity.hardVoting(inpuType='Single', model_list=model_list, images=images)
                pred = out
            else:
                model.eval()
                out = model(images)
                out = F.softmax(out, dim=1)
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
                    wrongCase_info = image_names[w_idx] + ' ' + str(out[w_idx, :].tolist()) + ' ' + str(
                        labels[w_idx].item())
                else:
                    wrongCase_info = image_names[w_idx] + ' ' + str(out[w_idx]) + ' ' + str(labels[w_idx].item())
                hard_examples.append(wrongCase_info)

    # 写入混淆矩阵
    dir_name = 'Baseline' if not ensemble_type else 'SingleInput'
    write_CM_HardExamples(runningOn, isCM=True, isHardExample=True, ensemble_type=ensemble_type, y_true=y_true,
                          y_pred=y_pred,
                          dir_name=dir_name,
                          model_name=model_name, dataset_name=dataset_name, hard_examples=hard_examples)


    # 输出正确率
    test_accuracy = num_correct / len(test_dataset)
    print(f'{model_name} accuracy is: {test_accuracy:.10f}')


def test_multipleInput(runningOn, test_dataset, test_loader, model_list=None, **kwargs):
    model_name = kwargs["model_name"]
    dataset_name = kwargs["dataset_name"]
    ensemble_type = kwargs["ensemble_type"]

    num_correct = 0
    hard_examples = []

    # 预测结果和真实标签
    y_pred = []
    y_true = []

    with torch.no_grad():
        for index, data in enumerate(tqdm(test_loader)):
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
                out = diversity.softVoting(inpuType='Multiple', model_list=model_list,
                                           images=images, model_outputs=[out1, out2, out3])
                # out = ensemble_models.multiple_input_soft_voting([out1, out2, out3])
                _, pred = torch.max(out, 1)
                pred = pred.to(DEVICE)
            else:
                out = diversity.hardVoting(inpuType='Multiple', model_list=model_list,
                                           images=images, model_outputs=[out1, out2, out3])
                # out = ensemble_models.multiple_input_hard_voting([out1, out2, out3])
                pred = out.to(DEVICE)

            # 将label和pred加入列表中
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

            num_correct += (pred == labels).sum()

            # 为混淆矩阵做准备
            all_index = torch.arange(start=0, end=image_num).to(DEVICE)
            wrong_idx = all_index[pred != labels]
            for w_idx in wrong_idx:
                if ensemble_type == 'soft':
                    wrongCase_info = image_names[w_idx] + ' ' + str(out[w_idx, :].tolist()) + ' ' + str(
                        labels[w_idx].item())
                else:
                    wrongCase_info = image_names[w_idx] + ' ' + str(out[w_idx]) + ' ' + str(labels[w_idx].item())
                hard_examples.append(wrongCase_info)


    # 写入混淆矩阵
    write_CM_HardExamples(runningOn, isCM=True, isHardExample=True, ensemble_type=ensemble_type, y_true=y_true,
                          y_pred=y_pred,
                          dir_name='MultipleInput',
                          model_name=model_name, dataset_name=dataset_name, hard_examples=hard_examples)

    # 输出正确率
    test_accuracy = num_correct / len(test_dataset)
    print(f'{model_name} accuracy is: {test_accuracy:.10f}')
