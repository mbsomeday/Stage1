import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils, datasets
from torchvision import transforms as T


from ww_model import vgg16_bn

mean = (0.4914, 0.4822, 0.4465)
std = (0.2471, 0.2435, 0.2616)
batch_size = 64

def test_dataloader():
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
    )
    return dataset, dataloader


# num_classes = 10
if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False


test_dataset, test_loader = test_dataloader()

model = vgg16_bn(pretrained=True)

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








































