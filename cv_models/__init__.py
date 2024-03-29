import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LOCAL = {
    # 模型权重
    'weights_path': {
        'CNN': r'D:\my_phd\on_git\experiment\data\model_weights\MyNet-054-0.9708.pth',
        'Inception': r'D:\my_phd\on_git\experiment\data\model_weights\Inception-043-0.99204081.pth',
        'ResNet': r'D:\my_phd\on_git\experiment\data\model_weights\ResNet-035-0.9952.pth',
        'VGG': r'D:\my_phd\on_git\experiment\data\model_weights\Vgg11-026-0.98806125.pth'
    },
    # 使用的常量
    'vars': {
        # 'base_dir': r'/content/SegData/data/completeData/left_images',
        # 'txt_dir': r'/content',
        
        'base_dir': r'D:\chrom_download\DaimlerPedestrianDetectionBenchmark\PedCut2013_SegmentationDataset\data\completeData\left_images',
        'txt_dir': 'D:\my_phd\on_git\experiment\data\dataset_txt\DaiPedSegmentation',
        'evaluation_dir': r'D:\my_phd\on_git\experiment\data\Model_Evaluation'
    }
}

CLOUD = {
    # 模型权重
    'weights_path': {
        'CNN': r'/content/drive/MyDrive/ColabNotebooks/data/model_weights/CNN/MyNet-054-0.9708.pth',
        'Inception': r'/content/drive/MyDrive/ColabNotebooks/data/model_weights/Inception/Inception-043-0.99204081.pth',
        'ResNet': r'/content/drive/MyDrive/ColabNotebooks/data/model_weights/ResNet/ResNet-035-0.9952.pth',
        'VGG': r'/content/drive/MyDrive/ColabNotebooks/data/model_weights/VGG/Vgg11-026-0.98806125.pth',
    },
    # 使用的常量
    'vars': {
        'base_dir': r'/content/images',
        'txt_dir': r'/content/drive/MyDrive/ColabNotebooks/data/dataset_txt',
        'evaluation_dir': r'/content/drive/MyDrive/ColabNotebooks/data/model_evaluation'
    }
}
