from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils, datasets
from torchvision import transforms as T


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
    dataset = datasets.CIFAR10(root=r'D:\my_phd\on_git\test\data', train=False, transform=transform, download=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
    )
    return dataset, dataloader

test_dataloader()
