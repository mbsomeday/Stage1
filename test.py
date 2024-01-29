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


def ensemble_test(args):
    BASE_DIR = r'images'
    TXT_DIR = r'/content/drive/MyDrive/ColabNotebooks/data/dataset_txt'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weight_path1 = None
    weight_path2 = None
    weight_path3 = None

    model1 = basic_learners.get_MyNet(device, pretrained=True, weights_path=weight_path1)
    model2 = basic_learners.get_ResNet(device, pretrained=True, weights_path=weight_path2)
    model3 = basic_learners.get_Inception(device, pretrained=True, weights_path=weight_path3)
    model_list = [model1, model2, model3]

    img_transformer = transforms.Compose([
        # transforms.Resize((36, 18)),   # (h, w)
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    test_dataset = MyDataset(base_dir=BASE_DIR, txt_dir=TXT_DIR, txt_name='test.txt', transform=img_transformer)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    print('Total training samples:', len(test_dataset))
    print('Total index samples:', len(test_loader))

    # 评估模型结果
    num_correct = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            out = ensemble_models.soft_voting(model_list, images)
            _, pred = torch.max(out, 1)
            num_correct += (pred == labels).sum()

        test_accuracy = num_correct / len(test_dataset)
        print('Test accuracy:{:.10f}'.format(test_accuracy))


def test_model_args(args):
    model_name = args.model_name
    weights_path = args.weights_path
    is_ensemble = args.is_ensemble
    MyNet_weight = args.MyNet_weight
    ResNet_weight = args.ResNet_weight
    Inception_weight = args.Inception_weight

    if is_ensemble:
        model1 = basic_learners.get_model('CNN', pretrained=True, weights_path=MyNet_weight)
        model2 = basic_learners.get_model('ResNet', pretrained=True, weights_path=ResNet_weight)
        model3 = basic_learners.get_model('Inception', pretrained=True, weights_path=Inception_weight)

        model_list = [model1, model2, model3]
    else:
        model = basic_learners.get_model(model_name, pretrained=True, weights_path=weights_path)
        model.eval()

    test_dataset, test_loader = dataset.get_dataloader(args)

    num_correct = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            if not is_ensemble:
                out = model(images)
            else:
                out = ensemble_models.soft_voting(model_list=model_list, images=images)
            _, pred = torch.max(out, 1)
            num_correct += (pred == labels).sum()

        test_accuracy = num_correct / len(test_dataset)
        print(f'{model_name} accuracy is "{test_accuracy:.10f}')


def test_model(test_dataset, test_loader, model, model_name, is_ensemble=False):
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
                out = ensemble_models.soft_voting(model_list=model, images=images)
            else:
                model.eval()
                out = model(images)
            _, pred = torch.max(out, 1)
            # 将label和pred加入列表中
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

            num_correct += (pred == labels).sum()

            # # 为混淆矩阵做准备
            # all_index = torch.arange(start=0, end=(images.shape[0]))
            # wrong_idx = all_index[pred != labels]
            # wrongCase_labels = labels[wrong_idx].item()
            # wrong_out = out[wrong_idx]
            # for w_idx in wrong_idx:
            #     wrongCase_info = image_names[w_idx] + ' ' + str(wrong_out) + ' ' + str(wrongCase_labels)
            #     hard_examples.append(wrongCase_info)

    # # 写入混淆矩阵
    # write_to_dir = r'/content/drive/MyDrive/ColabNotebooks/data/model_evaluation'
    # cm_name = f'{model_name}.csv'
    # cm_path = os.path.join(write_to_dir, 'Confusion_metrics', cm_name)
    # cm = confusion_matrix(y_true, y_pred)
    # pd.DataFrame(cm).to_csv(cm_path, index=False, header=False)
    # print('Confusion file written successfully!')

    # # 写入错分样本
    # hard_example_name = f'{model_name}.txt'
    # hard_example_path = os.path.join(write_to_dir, 'Hard_example_predictions', hard_example_name)
    # with open(hard_example_path, 'w') as f:
    #     for item in hard_examples:
    #         f.write(item + '\n')
    # print('Hard examples written successfully!')
    # print(len(hard_examples))

    # 输出正确率
    test_accuracy = num_correct / len(test_dataset)
    print(f'{model_name} accuracy is "{test_accuracy:.10f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--model_name', type=str,  choices=['CNN', 'Inception', 'ResNet', 'SoftVoting', 'HardVoting'], default="Model", required=True)

    parser.add_argument('--weights_path', type=str, required=False)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--txt_dir', type=str, required=True)
    parser.add_argument('--txt_name', type=str, choices=['train.txt', 'val.txt', 'test.txt'], required=True)
    parser.add_argument('--weights_path', type=str, required=True)


    args = parser.parse_args()

    image_dir = args.image_dir
    txt_dir = args.txt_dir
    txt_name = args.txt_name
    model_name = args.model_name
    weights_path = args.weights_path

    ret_dataset, ret_loader = dataset.get_dataloader(image_dir, txt_dir, txt_name, transformer_mode=1)

    test_model(ret_dataset, ret_loader, model_name)


