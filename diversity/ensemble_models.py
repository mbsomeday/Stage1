import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from cv_models import DEVICE, basic_learners


def single_input_hard_voting(model_list, images):
    out_list = []
    for model in model_list:
        model.eval()
        with torch.no_grad():
            out = model(images)
            _, pred = torch.max(out, 1)
            out_list.append(pred)

    temp = [out_list[i].cpu().numpy() for i in range(len(out_list))]
    res = np.stack(temp, axis=0)
    hard_res = res.sum(axis=0)
    hard_res[hard_res == 1] = 0
    hard_res[hard_res >= 2] = 1
    hard_res = torch.tensor(hard_res).to(DEVICE)

    return hard_res

def single_input_soft_voting(model_list, images):
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
    soft_res = torch.tensor(soft_res)

    return soft_res

def multiple_input_soft_voting(model_outputs):
    out_list = []
    for out in model_outputs:
        out = F.softmax(out, dim=1)
        out_list.append(out.cpu().numpy())

    temp = [out_list[i] for i in range(len(out_list))]
    res = np.stack(temp, axis=0)
    soft_res = np.sum(res, axis=0)
    soft_res = torch.tensor(soft_res)
    return soft_res

def multiple_input_hard_voting(model_outputs):
    out_list = []
    for out in model_outputs:
        _, pred = torch.max(out, 1)
        out_list.append(pred.cpu().numpy())

    temp = [out_list[i] for i in range(len(out_list))]
    res = np.stack(temp, axis=0)
    hard_res = res.sum(axis=0)
    hard_res[hard_res == 1] = 0
    hard_res[hard_res >= 2] = 1
    hard_res = torch.tensor(hard_res)

    return hard_res























