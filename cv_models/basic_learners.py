from torch import nn
import torch.nn.functional as F
import torch

from cv_models import DEVICE

# ==================================== CNN network ====================================


class MyNet(nn.Module):
    '''
      CNN network
      input_size: (18, 36)
    '''

    def __init__(self):
        super(MyNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, (6, 3)),
            nn.ReLU(),

            nn.Conv2d(6, 16, (12, 6)),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 11 * 20, 128),
            nn.ReLU(),

            nn.Linear(128, 512),
            nn.ReLU(),

            nn.Linear(512, 2)
        )

    def forward(self, x):
        # input size:(18, 36)
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


def get_MyNet(pretrained=False, weights_path=None):
    model = MyNet().to(DEVICE)
    if pretrained:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(DEVICE)))
        print('Loaded pretrained CNN.')
    else:
        print('Loaded un-pretrained CNN.')
    return model

# ==================================== ResNet ====================================

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (5, 3), 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, (5, 3), 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2_shortcut = nn.Sequential(
            nn.Conv2d(64, 64, (7, 3), 1, 1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, (8, 5), 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, (8, 5), 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, (11, 5), 1),
            nn.BatchNorm2d(128)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, (13, 7), 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (13, 7), 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv4_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, (21, 9), 1),
            nn.BatchNorm2d(256)
        )

        self.fc = nn.Linear(256 * 6 * 2, 2)

    def forward(self, x):
        conv1_res = self.conv1(x)

        conv2_res = self.conv2(conv1_res)
        conv2_shortcut_res = F.relu(conv2_res + self.conv2_shortcut(conv1_res))

        conv3_res = self.conv3(conv2_shortcut_res)
        conv3_shortcut_res = F.relu(conv3_res + self.conv3_shortcut(conv2_shortcut_res))

        conv4_res = self.conv4(conv3_shortcut_res)
        conv4_shortcut_res = F.relu(conv4_res + self.conv4_shortcut(conv3_shortcut_res))

        out = conv4_shortcut_res.view(x.size(0), -1)
        out = self.fc(out)

        return out

def get_ResNet(pretrained=False, weights_path=None):
    model = ResNet().to(DEVICE)
    if pretrained:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(DEVICE)))
        print('Loaded pretrained ResNet.')
    else:
        print('Loaded un-pretrained ResNet.')
    return model
# ==================================== Inception ====================================

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class Inception_Module(nn.Module):
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception_Module, self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv2d(in_c, c1, 1, 1, 0),
            nn.ReLU()
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(in_c, c2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=(5, 3), padding=(2, 1)),
            nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(in_c, c3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=(5, 3), padding=(2, 1)),
            nn.ReLU()
        )
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(5, 3), stride=1, padding=(2, 1)),
            nn.Conv2d(in_c, c4, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        return torch.cat((p1, p2, p3, p4), dim=1)


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv2d(1, 64, (5, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 128, (5, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.p2 = Inception_Module(128, 192, (96, 208), (16, 48), 64)
        self.p3 = Inception_Module(512, 256, (160, 320), (32, 128), 128)
        self.glob_pool = GlobalAvgPool2d()
        self.fc = nn.Linear(832, 2)

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.glob_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_Inception(pretrained=False, weights_path=None):
    model = Inception().to(DEVICE)
    if pretrained:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(DEVICE)))
        print('Loaded pretrained Inception.')
    else:
        print('Loaded un-pretrained Inception.')
    return model


# ==================================== 定义模型字典 ====================================

MODEL_NAME = {
    "CNN": MyNet,
    "Inception": Inception,
    "ResNet": ResNet
}


def get_model(model_name, pretrained=False, weights_path=None):

    model = MODEL_NAME.get(model_name)
    if pretrained:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(DEVICE)))
        print(f'Loaded pretrained {model_name}.')
    else:
        print(f'Loaded un-pretrained {model_name}.')
    return model



















