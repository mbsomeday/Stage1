from torchsummary import summary
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils, datasets
from torchvision import transforms as T
import numpy as np

from ww_model import vgg16_bn

grad_block = []	# 存放grad图
feaure_block = []	# 存放特征图
mean = (0.4914, 0.4822, 0.4465)
std = (0.2471, 0.2435, 0.2616)
batch_size = 1
H, W = 32, 32

def test_dataloader():
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    dataset = datasets.CIFAR10(root=r'D:\my_phd\on_git\test\data', train=False, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
    )
    return dataset, dataloader


# 获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 获取特征层的函数
def farward_hook(module, input, output):
    feaure_block.append(output)

# 已知原图、梯度、特征图，开始计算可视化图
def cam_show_img(img, feature_map, grads):
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 二维，用于叠加
    grads = grads.reshape([grads.shape[0], -1])
    # 梯度图中，每个通道计算均值得到一个值，作为对应特征图通道的权重
    weights = np.mean(grads, axis=1)
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]	# 特征图加权和
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    # cam.dim=2 heatmap.dim=3
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)	# 伪彩色
    # cam_img = 0.3 * heatmap + 0.7 * img

    # cam_img = cam_img.astype(np.uint8)

    # cv2.imshow('img', heatmap)
    # key = cv2.waitKey(0)

    # cv2.imwrite("cam.jpg", cam_img)



model = vgg16_bn()
print(model)
test_dataset, test_loader = test_dataloader()

model.features[40].register_forward_hook(farward_hook)
model.features[40].register_full_backward_hook(backward_hook)

i = 0
for data in test_loader:
    img, label = data
    output = model(img)
    max_idx = np.argmax(output.cpu().data.numpy())

    model.zero_grad()
    class_loss = output[0, max_idx]
    class_loss.backward()  # 反向梯度，得到梯度图

    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = feaure_block[0].cpu().data.numpy().squeeze()
    cam_show_img(img, fmap, grads_val)
    i += 1
    if i == 1:
        break













