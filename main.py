import argparse

from cv_models import basic_learners

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=["CNN", "Inception", "ResNet"], required=True)
    parser.add_argument('--pretrained', type=bool)
    parser.add_argument('--weights_path', type=str, required=False)

    args = parser.parse_args()
    model_name = args.model_name
    pretrained = args.pretrained
    print('pretrained:', pretrained)
    basic_learners.get_model(model_name, pretrained)











