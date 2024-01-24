import torch
import numpy as np

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

    out1 = out_list[0].cpu().numpy()
    out2 = out_list[1].cpu().numpy()
    out3 = out_list[2].cpu().numpy()

    res = np.stack((out1, out2, out3), axis=0)
    res_sum = res.sum(axis=0)

    res_sum[res_sum == 1] = 0
    res_sum[res_sum >= 2] = 1
    return res_sum


def model_ensembling(model_list, test_loader, test_dataset):
    num_correct = 0
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        out = hard_voting(model_list, images)
        out = torch.from_numpy(out).to(device)
        num_correct += (out == labels).sum()

    test_accuracy = num_correct / len(test_dataset)
    print('Voting test accuracy:', test_accuracy)

























