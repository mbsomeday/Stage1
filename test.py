from torchvision import transforms
import argparse
import torch
from torch.utils.data import DataLoader
import os

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

        model_list = [model1,model2, model3]
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

def test_model(model, test_dataset, test_loader):
    model.eval()
    num_correct = 0
    hard_examples = []
    with torch.no_grad():
        for index, data in enumerate(test_loader):
            images, labels, image_names = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            out = model(images)
            _, pred = torch.max(out, 1)
            num_correct += (pred == labels).sum()
            # print(pred != labels)
            all_index = torch.arange(start=0, end=(images.shape[0]))
            wrong_idx = all_index[pred != labels]
            # print('wrong idx:', wrong_idx)
            wrongCase_labels = labels[wrong_idx]
            # print('wrongCase_labels:', wrongCase_labels)
            wrong_out = out[wrong_idx]
            # print('wrong_out:', wrong_out)
            for w_idx in wrong_idx:
                wrongCase_info = image_names[w_idx] + ' ' + str(out[w_idx]) + ' ' + str(labels[w_idx].item())
                hard_examples.append(wrongCase_info)
            # print('hard_examples:', hard_examples)

        test_accuracy = num_correct / len(test_dataset)

        with open('CNN_wrongCase.txt', 'w') as f:
            for item in hard_examples:
                f.write(item + '\n')
        print(len(hard_examples))
        print(f'Model accuracy is "{test_accuracy:.10f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--model_name', type=str,  choices=['CNN', 'Inception', 'ResNet'], default="Model", required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--weighs_path', type=str, required=True)
    parser.add_argument('--is_ensemble', type=bool, required=True)
    parser.add_argument('--MyNet_weight', type=str, required=False)
    parser.add_argument('--ResNet_weight', type=str, required=False)
    parser.add_argument('--Inception_weight', type=str, required=False)

    args = parser.parse_args()

    test_single_model(args)

    # 将3个模型在新数据集上的结果保存为txt






















