import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils, datasets


from ww_model import vgg16_bn

batch_size = 64
# num_classes = 10
if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False

# 下载训练集 CIFAR-10训练集
# train_dataset = datasets.CIFAR10('./data', train=True, transform=transforms.ToTensor(), download=True)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = vgg16_bn()

# 测试开始
model.eval()
eval_loss = 0
eval_acc = 0

for data in test_loader:
    img, label = data
    if use_gpu:
        img = img.cuda()
        label = label.cuda()
    out = model(img)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss/len(test_dataset), eval_acc/len(test_dataset)))








































