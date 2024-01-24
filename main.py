import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from cv_models import basic_learners
from cv_models.ensemble_models import model_ensembling
from utils.dataset import MyDataset


parser = argparse.ArgumentParser(description='argparse testing')
parser.add_argument('--batch_size', type=int, default=64, required=True)

parser.add_argument('--CNN_weight', type=str, required=True)
parser.add_argument('--Inception_weight', type=str, required=True)
parser.add_argument('--ResNet_weight', type=str, required=True)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model1 = basic_learners.get_MyNet(device=device, weights_path=args.CNN_weight)
model2 = basic_learners.get_Inception(device=device, weights_path=args.Inception_weight)
model3 = basic_learners.get_ResNet(device=device, weights_path=args.ResNet_weight)

model_list = [model1, model2, model3]

BASE_DIR = r'images'
TXT_DIR = r'/content/drive/MyDrive/ColabNotebooks/data/dataset_txt'

BATCH_SIZE = args.batch_size

img_transformer = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = MyDataset(base_dir=BASE_DIR, txt_dir=TXT_DIR, txt_name='test.txt', transform=img_transformer)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


model_ensembling(model_list=model_list, test_loader=test_loader, test_dataset=test_dataset)








