import argparse
from torch.utils.data import DataLoader
from torchvision import transforms

from cv_models import DEVICE, basic_learners, LOCAL, CLOUD
from utils import dataset
from diversity import ensemble_models
import test
import train

if __name__ == '__main__':

    model_name = 'VGG'
    # weights_path = CLOUD_MODEL_WEIGHTS[model_name]
    weights_path = LOCAL_MODEL_WEIGHTS[model_name]
    model = basic_learners.get_model(model_name, pretrained=True, weights_path=weights_path)

    VARS = LOCAL_VARS

    # model_list = basic_learners.get_ensemble_model(model_name_list=['CNN', 'Inception', 'ResNet'], pretrained=True)


    # 用于单输入的
    train_dataset = dataset.MyDataset(image_dir=VARS['base_dir'], txt_dir=VARS['txt_dir'], txt_name='train.txt', transformer_mode=0)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)

    val_dataset = dataset.MyDataset(image_dir=VARS['base_dir'], txt_dir=VARS['txt_dir'], txt_name='val.txt', transformer_mode=0)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False)

    test_dataset = dataset.MyDataset(image_dir=VARS['base_dir'], txt_dir=VARS['txt_dir'], txt_name='test.txt', transformer_mode=0)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

    # ---------------------------- 模型训练 ----------------------------

    # # 模型训练
    # train.main(model=model, model_name='Vgg11', train_dataset=train_dataset, train_loader=train_dataloader,
    #            val_dataset=val_dataset, val_loader=val_dataloader, weight_save_path=r'')


    # ---------------------------- 模型测试 ----------------------------
    # 1.单输入
    test.test_model(test_dataset=test_dataset, test_loader=test_dataloader, model=model, model_name=model_name,
                    dataset_name="DaiPedClassify",
                    is_ensemble=False, ensemble_type='soft'
                    )

    # 2.多输入
    # test_dataset = dataset.Dataset_for_multiInput(base_dir=BASE_DIR, txt_dir=TXT_DIR, txt_name=txt_name,
    #                                               transform_list=transform_list)
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

    # test.multipleInput_voting(test_dataset=test_dataset, test_loader=test_dataloader, model_list=model_list,
    #                           ensemble_type='soft')






























