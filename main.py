from torch.utils.data import DataLoader

from cv_models import get_model, LOCAL
from utils import dataset
import test


model_name = 'VGG'
model = get_model.get_single_model(model_name, running_on=LOCAL, pretrained=True)

# model_name = 'CNNIncVGG'
# model_list = get_model.get_ensemble_model(['Inception', 'CNN', 'VGG'], running_on=LOCAL, pretrained=True)

# 单输入的dataset
test_dataset = dataset.MyDataset(running_on=LOCAL, txt_name='test.txt', transformer_mode=0)

# # 多输入的dataset
# test_dataset = dataset.MyDataset(running_on=LOCAL, txt_name='test.txt', multinput=True)

test_loader = DataLoader(test_dataset, 64, shuffle=False)

test.test_singleInput(runningOn=LOCAL, test_dataset=test_dataset, test_loader=test_loader,
                      model=model,
                      # model_list=model_list,
                      model_name=model_name, dataset_name='DaiPedSegmentation', ensemble_type=None
                      )
#DaiPedClassification
# test.test_multipleInput(runningOn=LOCAL, test_dataset=test_dataset, test_loader=test_loader, model_list=model_list,
#                         model_name=model_name, dataset_name='DaiPedSegmentation', ensemble_type='hard'
#                         )


