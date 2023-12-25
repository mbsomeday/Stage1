# https://blog.csdn.net/weixin_41735859/article/details/106474768

from train import load_model, get_dataloader
from torchvision.models import resnet50
import numpy as np
import cv2, os
import matplotlib.pylab as plt


fmap_block = list()
grad_block = list()


# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)


# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, out_dir='.'):
    C, H, W = img.shape
    cam = np.zeros(feature_map.shape[1: ], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.maximum(cam, 0)
    print(cam)
    print('-' * 100)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # cam_img = 0.3 * heatmap + 0.7 * img

    path_cam_img = os.path.join(out_dir, "cam.jpg")
    # cv2.imwrite(path_cam_img, cam_img)
    plt.subplot(1, 2, 1)
    plt.imshow(heatmap)
    plt.subplot(1, 2, 2)
    plt.imshow(img[0])
    plt.show()
    print(cam.shape)


model = load_model()
model.eval()

model.features[-1].register_forward_hook(farward_hook)
model.features[-1].register_backward_hook(backward_hook)


# model = resnet50(pretrained=True)
_, test_loader = get_dataloader()

for X, y in test_loader:
    img = X[0]
    out = model(img)
    idx = np.argmax(out.cpu().data.numpy())

    model.zero_grad()
    class_loss = out[0, idx]
    class_loss.backward()

    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    cam_show_img(img, fmap, grads_val)
    break











