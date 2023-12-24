from torch import nn
from torchsummary import summary
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F


BATCH_SIZE = 64
EPOCHS = 100


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(10, 20, 3),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, self.num_classes),
            # nn.ReLU(),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        out = torch.nn.functional.log_softmax(x, dim=1)
        return out

def train(dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        optimizer.zero_grad()
        pred = model(X)

        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])


train_set = datasets.MNIST(root=r'D:\my_phd\on_git\test\data',
                           train=True,
                           transform=transform,
                           download=False)
test_set = datasets.MNIST(root=r'D:\my_phd\on_git\test\data',
                          train=False,
                          transform=transform,
                          download=False)

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
model = LeNet(10)

# print(model)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# summary(model, (1, 28, 28), 4)

def test(dataloader):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    if correct > best_acc:
        torch.save(model.state_dict(), 'model.pth')

    print(f"Test accuracy: {(100*correct):>0.1f}%")


best_acc = 1e-10
for t in range(EPOCHS):
    print(f"-------------- {t+1} --------------")
    train(train_dataloader, loss, optimizer)
    test(test_dataloader)











