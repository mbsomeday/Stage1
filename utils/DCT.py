import cv2
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import stats
import matplotlib.pylab as plt



BATCH_SIZE = 1


def get_MNIST_DCT():
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_set = datasets.MNIST(root=r'data', train=True, download=False, transform=transformer)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    DCT_list = list()

    for X, y in tqdm(test_loader):
        image = X[0].numpy()
        image = np.transpose(image, (1, 2, 0))
        gray_image = np.float32(image)  # 将数值精度调整为32位浮点型
        img_dct = cv2.dct(gray_image)  # 使用dct获得img的频域图像
        DCT_list.append(img_dct[0][0])

    dct_array = np.array(DCT_list)
    np.save('MNIST_train_dct.npy', dct_array)

def get_CIFAR10_DCT():
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_set = datasets.CIFAR10(root=r'data', train=False, download=False, transform=transformer)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    DCT_list = list()

    for X, y in tqdm(test_loader):
        image = X[0].numpy()
        image = np.transpose(image, (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = np.float32(image)  # 将数值精度调整为32位浮点型
        img_dct = cv2.dct(gray_image)  # 使用dct获得img的频域图像
        DCT_list.append(img_dct[0][0])

    dct_array = np.array(DCT_list)
    np.save('CIFAR10_test_dct.npy', dct_array)


def get_MNIST_DCT_Byclass(save_path=r'DCT_npy/MNIST_train_9.npy'):
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_set = datasets.MNIST(root=r'../data', train=True, download=False, transform=transformer)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    DCT_list = list()

    for X, y in tqdm(test_loader):
        if y.item() == 9:
            image = X[0].numpy()
            image = np.transpose(image, (1, 2, 0))
            gray_image = np.float32(image)  # 将数值精度调整为32位浮点型
            img_dct = cv2.dct(gray_image)  # 使用dct获得img的频域图像
            DCT_list.append(img_dct[0][0])

    dct_array = np.array(DCT_list)
    # print(dct_array.shape)
    np.save(save_path, dct_array)


# get_MNIST_DCT_Byclass()

# X = np.zeros(shape=(10, 100))
# y = []
# for i in range(100):
#     y.append(i)
#
# for i in range(10):
#     dct_path = r'DCT_npy/MNIST_train_' + str(i) + '.npy'
#     current_DCT = np.load(dct_path)
#     X[i] = current_DCT[:100]
#
# z = [4.947853, -7.390806, 1.2366639, 4.2291627, -3.1332111, 2.0930922]
# for i in range(10):
#     plt.subplot(2, 5, int(i + 1))
#     plt.scatter(X[i], y, edgecolors='green', s=1)
#     plt.scatter(z[0], 20, label="0")
#     plt.scatter(z[1], 30, label="1")
#     plt.scatter(z[2], 40, label="2")
#     plt.scatter(z[3], 50, label="3")
#     plt.scatter(z[4], 60, label="4")
#     plt.scatter(z[5], 70, label="5")
#
#     x_max = X[i].max()
#     x_min = X[i].min()
#     plt.xlim(x_min-1, x_max+1)
#     plt.ylim(-10, 110)
#     plt.xlabel(str(i))
#     # plt.ylabel()
#
# # plt.suptitle('Test Case:label 5 [4.947853] VS DCT of All Classes')
#
# plt.legend()
#
# plt.tight_layout()
# plt.savefig('./All.jpg')
# plt.show()



# x = np.load('DCT_npy/MNIST_train_1.npy')
# x = x[:100]

mnist_train_0 = np.load('DCT_npy/MNIST_train_0.npy')
mnist_train_1 = np.load('DCT_npy/MNIST_train_1.npy')

mnist_train_0 = mnist_train_0[:100]
mnist_train_1 = mnist_train_1[:100]

res4 = stats.kstest(mnist_train_0, mnist_train_1)
print(f'mnist_train_0[{len(mnist_train_0)}] VS mnist_train_1[{len(mnist_train_1)}]:\n', res4)

# mnist_train = np.load('DCT_npy/MNIST_train_dct.npy')
# mnist_test = np.load('DCT_npy/MNIST_test_dct.npy')
# cifar10_test = np.load('DCT_npy/CIFAR10_test_dct.npy')

# mnist_train = mnist_train[:1000]
# mnist_test = mnist_test[:1000]
# cifar10_test = cifar10_test[:1000]

# res1 = stats.kstest(mnist_train, mnist_test)
# res2 = stats.kstest(mnist_test, cifar10_test)
# res3 = stats.kstest(mnist_train, cifar10_test)
#
# print(f'mnist_train[{len(mnist_train)}] VS mnist_test[{len(mnist_test)}]:\n', res1)
# print()
# print(f'mnist_test[{len(mnist_test)}] VS cifar10_test[{len(cifar10_test)}]:\n', res2)
# print()
# print(f'mnist_train[{len(mnist_train)}] VS cifar10_test[{len(cifar10_test)}]:\n', res3)





















