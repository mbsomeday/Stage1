import torch

from cv_models import DEVICE, basic_learners, vgg_model


MODEL_NAME = {
    "CNN": basic_learners.MyNet,
    "Inception": basic_learners.Inception,
    "ResNet": basic_learners.ResNet,
    "VGG": vgg_model.vgg11
}

# ==================================== 获取 single model ====================================

def get_single_model(model_name, running_on, pretrained=False):
    model = MODEL_NAME.get(model_name)().to(DEVICE)
    if pretrained:
        weights_path = running_on['weights_path'][model_name]
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(DEVICE)))
        print(f'Loaded pretrained ** {model_name} ** successfully!')
    else:
        print(f'Loaded un-pretrained ** {model_name} ** successfully!')
    return model

# ==================================== 获取ensemble model ====================================

def get_ensemble_model(model_name_list, running_on, pretrained=False):
    print('-' * 30 + 'Start Loading Ensemble Models!' +'-' * 30)
    model_list = []
    for name in model_name_list:
        model_list.append(get_single_model(name, running_on, pretrained=pretrained))
    return model_list










