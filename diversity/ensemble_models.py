import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from cv_models import basic_learners

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model1 = basic_learners.MyNet()
model2 = basic_learners.Inception()
model3 = basic_learners.ResNet()


def hard_voting(model_list, images):
    out_list = []
    for model in model_list:
        model.eval()
        with torch.no_grad():
            out = model(images)

            _, pred = torch.max(out, 1)
            out_list.append(pred)

    temp = [out_list[i] for i in range(len(out_list))]
    res = np.stack(temp, axis=0)

    res_sum = res.sum(axis=0)

    res_sum[res_sum == 1] = 0
    res_sum[res_sum >= 2] = 1
    return res_sum


def model_ensembling(model_list, test_loader, test_dataset):
    num_correct = 0
    for data in tqdm(test_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        out = hard_voting(model_list, images)
        out = torch.from_numpy(out).to(device)
        num_correct += (out == labels).sum()

    test_accuracy = num_correct / len(test_dataset)
    print('num_correct:', num_correct)
    print('total testing examples:', len(test_dataset))
    print('Hard voting test accuracy:', test_accuracy)

def soft_voting(model_list, images):
    out_list = []
    for model in model_list:
        model.eval()
        with torch.no_grad():
            out = model(images)
            # 对模型结果进行softmax，不然有负数出现
            out = F.softmax(out, dim=1)
            out_list.append(out.cpu().numpy())

    temp = [out_list[i] for i in range(len(out_list))]
    res = np.stack(temp, axis=0)
    soft_res = np.sum(res, axis=0)

    return soft_res


if __name__ == '__main__':
    print('-' * 30 + 'ensemble models' + '-' * 30)
    from cv_models import basic_learners

    torch.manual_seed(13)

    model1 = basic_learners.MyNet()
    model2 = basic_learners.ResNet()
    model3 = basic_learners.Inception()

    model_list = [model1, model2, model3]
    images = torch.rand(size=(5, 1, 36, 18))

    softres = soft_voting(model_list, images)




















