import torch
from torchvision import transforms
from torchsummary import summary
from torch import nn

from cv_models.VGG.ww_model import vgg16_bn
from utils import util

BASE_DIR = r'D:\chrom_download\DaimlerPedestrianDetectionBenchmark\DC-ped-dataset_base'
EPOCHS = 1
BATCH_SIZE = 8

transforms = transforms.Compose([
transforms.CenterCrop(64),
    transforms.ToTensor()
])

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*13*13, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 2)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

def main():
    model = MyNet()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_dataset = util.MyDataset(base_dir=BASE_DIR, txt_name='train.txt', transform=transforms)
    train_loader = util.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = util.MyDataset(base_dir=BASE_DIR, txt_name='val.txt', transform=transforms)
    val_loader = util.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        running_loss = 0.0

        # training
        for idx, data in enumerate(train_loader):
            model.train()

            images, labels = data
            out = model(images)

            loss = loss_fn(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (idx + 1) % 10 == 0:
                print('[%d  %5d]   loss: %.3f'%(epoch+1, idx+1, running_loss/100))
                running_loss = 0.0
            break

    # validation after finish EPOCHS training
    val_loss = 0.0
    for data in val_loader:
        model.eval()
        images, labels = data
        out = model(images)
        loss = loss_fn(out, labels)
        _, pred = torch.max(out, 1)
        num_correct = (pred == labels).sum()

        val_loss += loss.item()
    print('Val Loss:{:.6f}, accuracy:{:.6f}'.format(val_loss, num_correct/len(val_dataset)))


if __name__ == '__main__':
    main()










