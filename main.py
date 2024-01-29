import argparse
from torch.utils.data import DataLoader
from torchvision import transforms

from cv_models import DEVICE, basic_learners
from utils import dataset
from diversity import ensemble_models
import test


if __name__ == '__main__':
    model_name = 'Ensemble'
    CNN_weights_path = r'D:\my_phd\on_git\experiment\data\model_weights\MyNet-054-0.9708.pth'
    Inception_weights_path = r'D:\my_phd\on_git\experiment\data\model_weights\Inception-043-0.99204081.pth'
    ResNet_weights_path = r'D:\my_phd\on_git\experiment\data\model_weights\ResNet-035-0.9952.pth'

    model_list = basic_learners.get_ensemble_model(pretrained=True, CNN_weights_path=CNN_weights_path,
                                              Inception_weights_path=Inception_weights_path,
                                              ResNet_weights_path=ResNet_weights_path)

    # model = basic_learners.get_model(model_name=model_name, pretrained=True, weights_path=CNN_weights_path)

    image_transform2 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1),  # 水平翻转
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.ToTensor()
    ])

    test_dataset = dataset.Fake_dataset(transform2=image_transform2, transform3=image_transform2)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, drop_last=False)


    # test.test_model(test_dataset=test_dataset, test_loader=test_dataloader, model=model, model_name=model_name,
    #                 is_ensemble=True, ensemble_type='hard')

    test.multipleInput_voting(test_dataset=test_dataset, test_loader=test_dataloader, model_list=model_list, ensemble_type='soft')






















