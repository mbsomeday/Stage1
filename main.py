import argparse
from torch.utils.data import DataLoader
from torchvision import transforms

from cv_models import DEVICE, basic_learners, CLOUD_MODEL_WEIGHTS, BASE_DIR, TXT_DIR, vgg_model
from utils import dataset
from diversity import ensemble_models
import test
import train

if __name__ == '__main__':

    model = vgg_model.vgg11()
    model = model.to(DEVICE)
    model_name = 'VGG11'
    txt_name = 'test.txt'

    # weights_path = CLOUD_MODEL_WEIGHTS[model_name]

    # model = basic_learners.get_model(model_name=model_name, pretrained=False, weights_path=weights_path)
    model_list = basic_learners.get_ensemble_model(pretrained=True, CNN_weights_path=CLOUD_MODEL_WEIGHTS['CNN'],
                                                   Inception_weights_path=CLOUD_MODEL_WEIGHTS['Inception'],
                                                   ResNet_weights_path=CLOUD_MODEL_WEIGHTS['ResNet'])

    # transform_list = [image_transform1, image_transform2, image_transform3]
    # print('number of models:', len(model_list))

    # 用于单输入的
    train_dataset = dataset.MyDataset(image_dir=BASE_DIR, txt_dir=TXT_DIR, txt_name='train.txt', transformer_mode=1)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)

    val_dataset = dataset.MyDataset(image_dir=BASE_DIR, txt_dir=TXT_DIR, txt_name='val.txt', transformer_mode=1)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False)

    test_dataset = dataset.MyDataset(image_dir=BASE_DIR, txt_dir=TXT_DIR, txt_name='test.txt', transformer_mode=1)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

    # # 模型训练
    # train.main(model=model, model_name='Vgg11', train_dataset=train_dataset, train_loader=train_dataloader,
    #            val_dataset=val_dataset, val_loader=val_dataloader, weight_save_path=r'')

    # # 用于测试的dataset
    # test_dataset = dataset.Fake_dataset()
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)


    # 用于多输入的
    # test_dataset = dataset.Dataset_for_multiInput(base_dir=BASE_DIR, txt_dir=TXT_DIR, txt_name=txt_name,
    #                                               transform_list=transform_list)
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

    # 单输入 多模型测试
    test.test_model(test_dataset=test_dataset, test_loader=test_dataloader, model=model_list,
                    dataset_name="DaiPedClassify", is_ensemble=True, ensemble_type='hard')

    # test.multipleInput_voting(test_dataset=test_dataset, test_loader=test_dataloader, model_list=model_list,
    #                           ensemble_type='soft')

