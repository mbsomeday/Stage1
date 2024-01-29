import argparse
from torch.utils.data import DataLoader
from torchvision import transforms

from cv_models import DEVICE, basic_learners, CLOUD_MODEL_WEIGHTS, BASE_DIR, TXT_DIR
from utils import dataset
from diversity import ensemble_models
import test

if __name__ == '__main__':
    model_name = 'CNN'
    CNN_weights_path = CLOUD_MODEL_WEIGHTS['CNN']
    Inception_weights_path = CLOUD_MODEL_WEIGHTS['Inception']
    ResNet_weights_path = CLOUD_MODEL_WEIGHTS['ResNet']
    txt_name = 'test.txt'

    weights_path = CNN_weights_path

    model_list = basic_learners.get_ensemble_model(pretrained=True, CNN_weights_path=CNN_weights_path,
                                                   Inception_weights_path=Inception_weights_path,
                                                   ResNet_weights_path=ResNet_weights_path)

    # model = basic_learners.get_model(model_name=model_name, pretrained=True, weights_path=CNN_weights_path)

    image_transform1 = transforms.Compose([
        transforms.ToTensor()
    ])

    image_transform2 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1),  # 水平翻转
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.ToTensor()
    ])

    image_transform3 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
        transforms.ToTensor()
    ])

    transform_list = [image_transform1, image_transform2, image_transform3]
    # print('number of models:', len(model_list))

    # test_dataset = dataset.Fake_dataset(transform2=image_transform2, transform3=image_transform2)
    # test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, drop_last=False)

    test_dataset = dataset.Dataset_for_multiInput(base_dir=BASE_DIR, txt_dir=TXT_DIR, txt_name=txt_name,
                                                  transform_list=transform_list)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

    # test.test_model(test_dataset=test_dataset, test_loader=test_dataloader, model=model, model_name=model_name,
    #                 is_ensemble=True, ensemble_type='hard')

    test.multipleInput_voting(test_dataset=test_dataset, test_loader=test_dataloader, model_list=model_list,
                              ensemble_type='soft')

    dataset.MyDataset(base_dir=BASE_DIR, txt_dir=TXT_DIR, txt_name=txt_name,transform_list=transform_list)