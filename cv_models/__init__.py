import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LOCAL_MODEL_WEIGHTS = {
    'CNN': r'D:\my_phd\on_git\experiment\data\model_weights\MyNet-054-0.9708.pth',
    'Inception': r'D:\my_phd\on_git\experiment\data\model_weights\Inception-043-0.99204081.pth',
    'ResNet': r'D:\my_phd\on_git\experiment\data\model_weights\ResNet-035-0.9952.pth'
}

CLOUD_MODEL_WEIGHTS = {
    'CNN': r'/content/drive/MyDrive/ColabNotebooks/data/model_weights/CNN/MyNet-054-0.9708.pth',
    'Inception': r'/content/drive/MyDrive/ColabNotebooks/data/model_weights/Inception/Inception-043-0.99204081.pth',
    'ResNet': r'/content/drive/MyDrive/ColabNotebooks/data/model_weights/ResNet/ResNet-035-0.9952.pth'
}

BASE_DIR = r'/content/images'
TXT_DIR = r'/content/drive/MyDrive/ColabNotebooks/data/dataset_txt'