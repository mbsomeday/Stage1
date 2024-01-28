import argparse

from cv_models import basic_learners
from utils import dataset
import test

if __name__ == '__main__':
    # 将pre-trained model错误的结果保存
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=["CNN", "Inception", "ResNet"], required=True)
    parser.add_argument('--pretrained', type=bool)
    parser.add_argument('--weights_path', type=str, required=False)

    args = parser.parse_args()
    model_name = args.model_name
    weights_path = args.weights_path
    image_dir = args.image_dir
    txt_dir = args.txt_dir
    txt_name = args.txt_name

    model = basic_learners.get_model(model_name=model_name, pretrained=True, weights_path=weights_path)
    ret_dataset, ret_loader = dataset.get_dataloader(image_dir, txt_dir, txt_name, transformer_mode=1)
    test.test_model(model, ret_dataset, ret_loader)












