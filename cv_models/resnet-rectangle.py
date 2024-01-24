import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ResNet(nn.Module):
    '''
        input size: (18, 36)
    '''
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 5), 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, (3, 5), 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2_shortcut = nn.Sequential(
            nn.Conv2d(64, 64, (3, 7), 1, 1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 8), 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, (5, 8), 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, (5, 11), 1),
            nn.BatchNorm2d(128)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, (7, 13), 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (7, 13), 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv4_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, (9, 21), 1),
            nn.BatchNorm2d(256)
        )

        self.fc = nn.Linear(256*6*2, 2)


    def forward(self, x):
        conv1_res = self.conv1(x)

        conv2_res = self.conv2(conv1_res)
        conv2_shortcut_res = F.relu(conv2_res + self.conv2_shortcut(conv1_res))

        conv3_res = self.conv3(conv2_shortcut_res)
        conv3_shortcut_res = F.relu(conv3_res + self.conv3_shortcut(conv2_shortcut_res))

        conv4_res = self.conv4(conv3_shortcut_res)
        conv4_shortcut_res = F.relu(conv4_res + self.conv4_shortcut(conv3_shortcut_res))

        out = conv4_shortcut_res.view(x.size(0), -1)
        print(out.shape)
        out = self.fc(out)

        return out
























