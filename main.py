import argparse
from torch.utils.data import DataLoader

from cv_models import DEVICE, basic_learners
from utils import dataset
from diversity import ensemble_models
import test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=False)


    args = parser.parse_args()

    model_name = args.model_name
    CNN_weights_path = r'D:\my_phd\on_git\experiment\data\model_weights\MyNet-054-0.9708.pth'
    Inception_weights_path = r'D:\my_phd\on_git\experiment\data\model_weights\Inception-043-0.99204081.pth'
    ResNet_weights_path = r'D:\my_phd\on_git\experiment\data\model_weights\ResNet-035-0.9952.pth'

    model = basic_learners.get_ensemble_model(pretrained=True, CNN_weights_path=CNN_weights_path,
                                              Inception_weights_path=Inception_weights_path,
                                              ResNet_weights_path=ResNet_weights_path)
    # model = basic_learners.get_model(model_name=model_name, pretrained=True, weights_path=CNN_weights_path)

    test_dataset = dataset.Fake_dataset()
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, drop_last=False)

    test.test_model(test_dataset=test_dataset, test_loader=test_dataloader, model=model, model_name='Ensemble',
                    is_ensemble=True)






















