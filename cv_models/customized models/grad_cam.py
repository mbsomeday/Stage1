# https://blog.csdn.net/weixin_41735859/article/details/106474768
import torch

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
    grads = grads.reshape([grads.shape[0], -1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    ww_mask = (cam > 1e-10) + 0.0
    ww_mask = torch.Tensor(ww_mask).unsqueeze(0).numpy()     # ww_mask.shape: (1, 28, 28)
    ww_mask *= 255
    ww_mask = ww_mask.astype(np.uint8)

    # print('mask：', ww_mask)

    # heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    ww_img = (img * 255).numpy().astype(np.uint8)
    # print('img:\n', ww_img)
    ww_result = cv2.bitwise_and(ww_img, ww_mask)   # ww_result.shape:(1, 28, 28)

    # print('与运算后的result：\n', ww_result)
    print(ww_img.shape)

    ww_result = np.transpose(ww_result, (1, 2, 0))
    #
    # print('img.shape', img[0].shape)
    # print('heatmap', heatmap.shape)
    # print(ww_result.shape)

    # cam_img = 0.3 * heatmap + 0.7 * img

    #  note: plt image.shape: h w c
    # path_cam_img = os.path.join(out_dir, "cam.jpg")
    # cv2.imwrite(path_cam_img, cam_img)

    plt.subplot(1, 3, 1)
    plt.title('img')
    plt.imshow(np.transpose(ww_img, (1, 2, 0)))

    plt.subplot(1, 3, 2)
    plt.title('cam')
    plt.imshow(cam)

    plt.subplot(1, 3, 3)
    plt.title('ROI')
    plt.imshow(ww_result)
    plt.show()

    # print('0000', np.transpose(ww_result, (2, 1, 0)))


model = load_model()
model.eval()

model.features[-1].register_forward_hook(farward_hook)
model.features[-1].register_backward_hook(backward_hook)

_, test_loader = get_dataloader(True)

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











