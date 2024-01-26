from torchvision import transforms
import argparse
import torch
from torch.utils.data import DataLoader
import os

from diversity import ensemble_models
from cv_models import basic_learners
from utils.dataset import MyDataset


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--weight_path', type=str, required=True, help='change weight path in code')

    args = parser.parse_args()

    ensemble_test(args)




















