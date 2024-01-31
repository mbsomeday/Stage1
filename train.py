from torchvision import transforms
import argparse
import torch
from torch.utils.data import DataLoader
import os

from cv_models import basic_learners, DEVICE
from utils.dataset import MyDataset


def main(model, model_name, train_dataset, train_loader,
         val_dataset, val_loader,
         weight_save_path):


    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    EPOCHS = 50
    BATCH_SIZE = 64
    BEST_ACCURACY = -10.0

    WEIGHT_SAVE_PATH = weight_save_path


    print('Total training samples:', len(train_dataset))
    print('Total index samples:', len(train_loader))

    for epoch in range(EPOCHS):

        # training
        model.train()
        running_loss = 0.0
        for batch, data in enumerate(train_loader):
            images, labels, _ = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            out = model(images)

            loss = loss_fn(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch + 1) % 100 == 0:
                print('Training Epoch: %d, batch_idx:%5d, loss: %.8f' % (epoch + 1, batch + 1, running_loss / 100))
                running_loss = 0.0

        # validation after finish One EPOCH training
        model.eval()
        val_loss = 0.0
        num_correct = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels, _ = data
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                out = model(images)
                loss = loss_fn(out, labels)
                _, pred = torch.max(out, 1)
                num_correct += (pred == labels).sum()
                val_loss += loss.item()

            val_accuracy = num_correct / len(val_dataset)
            print('Val Loss:{:.6f}, accuracy:{:.10f}'.format(val_loss, val_accuracy))
            print('*' * 50)
        # 如果模型表现效果好，则保存
        if val_accuracy > BEST_ACCURACY:
            model_save_dir = WEIGHT_SAVE_PATH

            # 删除已经有的文件,只保留n+1个模型
            num_saved = 2
            all_weights = os.listdir(model_save_dir)
            if len(all_weights) > num_saved:
                sorted = []
                for weight in all_weights:
                    acc = weight.split('-')[-1]
                    sorted.append((weight, acc))
                sorted.sort(key=lambda w: w[1], reverse=True)
                del_path = os.path.join(model_save_dir, sorted[-1][0])
                os.remove(del_path)
                print('del file:', del_path)

            # 保存新的模型
            save_name = f"{model_name}-{epoch + 1:03d}-{val_accuracy:.8f}.pth"
            save_path = os.path.join(model_save_dir, save_name)
            # 用ckpt模式保存权重，总是有问题，然后换新方式存储权重
            # checkpoint = {'model': model.state_dict(), 'epoch': epoch, 'val_accuracy':val_accuracy}
            # torch.save(checkpoint , save_path)

            # 新方式存储权重
            torch.save(model.state_dict(), save_path)

            BEST_ACCURACY = val_accuracy
            print(f'{model_name} saved on epoch:{epoch + 1}, current BEST_ACCURACY:{BEST_ACCURACY}')
            print('*' * 50)


if __name__ == '__main__':
    from cv_models import vgg_model

    model = vgg_model.vgg11()


