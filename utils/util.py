from sklearn.metrics import confusion_matrix
import pandas as pd
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 获取混淆矩阵并保存
def get_cm(model, test_loader, write_to_file):
    model.eval()
    # 预测结果和真实标签
    y_pred = []
    y_true = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm).to_csv(write_to_file, index=False, header=False)
    return cm


# 根据混淆矩阵计算某类的召回率
# https://blog.csdn.net/a645676/article/details/127369713
def calculate_label_recall(confMatrix, model_name):
    l = len(confMatrix)
    for i in range(l):
        label_total_sum = confMatrix.sum(axis=0)[i]
        label_correct_sum = confMatrix[i][i]
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
        print(f'{model_name} recall of class {i}: {str(prediction)}%')
