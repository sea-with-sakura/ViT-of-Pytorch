import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import transforms


__all__ = ['CIFAR10DataLoader', 'ImageNetDataLoader', 'CIFAR100DataLoader', 'TinyImageNetDataLoader', 'set_seed']


def set_seed(seed=42):
    """
    设置随机种子以确保实验可重复性
    
    参数:
        seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保CUDA操作是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)


class CIFAR10DataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8, seed=42):
        if split == 'train':
            train = True
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.dataset = CIFAR10(root=data_dir, train=train, transform=transform, download=True)

        # 使用固定的随机种子生成器
        generator = torch.Generator()
        generator.manual_seed(seed)

        super(CIFAR10DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False if not train else True,
            num_workers=num_workers,
            generator=generator)


class CIFAR100DataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8, seed=42):
        if split == 'train':
            train = True
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.dataset = CIFAR100(root=data_dir, train=train, transform=transform, download=True)

        # 使用固定的随机种子生成器
        generator = torch.Generator()
        generator.manual_seed(seed)

        super(CIFAR100DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False if not train else True,
            num_workers=num_workers,
            generator=generator)


class TinyImageNetDataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=64, batch_size=16, num_workers=8, seed=42):
        """
        Tiny ImageNet 数据加载器
        
        Args:
            data_dir (str): 数据集根目录路径
            split (str): 数据集划分，'train', 'val' 或 'test'
            image_size (int): 图像大小，默认为 64 (Tiny ImageNet 原始尺寸)
            batch_size (int): 批次大小
            num_workers (int): 数据加载的工作进程数
            seed (int): 随机种子
        """
        # Tiny ImageNet 的标准变换
        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:  # val or test
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        # Tiny ImageNet 数据集路径结构
        if split == 'train':
            # 训练集路径: data_dir/train/
            dataset_path = os.path.join(data_dir, 'train')
        elif split == 'val':
            # 验证集路径: data_dir/val/
            dataset_path = os.path.join(data_dir, 'val')
        else:  # test
            # 测试集路径: data_dir/test/
            dataset_path = os.path.join(data_dir, 'test')
            
        # 处理带有images子文件夹的数据结构
        samples = []
        targets = []
        class_to_idx = {}
        
        # 获取所有类别文件夹
        classes = sorted(os.listdir(dataset_path))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        # 遍历每个类别文件夹及其images子文件夹
        for cls_name in classes:
            cls_idx = class_to_idx[cls_name]
            cls_path = os.path.join(dataset_path, cls_name, 'images')
            if os.path.exists(cls_path):
                for img_name in os.listdir(cls_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        samples.append(os.path.join(cls_path, img_name))
                        targets.append(cls_idx)
        
        # 创建自定义数据集
        from torch.utils.data import Dataset
        class CustomDataset(Dataset):
            def __init__(self, samples, targets, transform):
                self.samples = samples
                self.targets = targets
                self.transform = transform
                
            def __len__(self):
                return len(self.samples)
                
            def __getitem__(self, idx):
                from PIL import Image
                img_path = self.samples[idx]
                target = self.targets[idx]
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, target
                
        self.dataset = CustomDataset(samples, targets, transform)
        
        # 使用固定的随机种子生成器
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        super(TinyImageNetDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=num_workers,
            generator=generator)


class ImageNetDataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8, seed=42):

        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.dataset = ImageFolder(root=os.path.join(data_dir, split), transform=transform)
        
        # 使用固定的随机种子生成器
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        super(ImageNetDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=num_workers,
            generator=generator)


if __name__ == '__main__':
    data_loader = ImageNetDataLoader(
        data_dir='/home/hchen/Projects/vat_contrast/data/ImageNet',
        split='val',
        image_size=384,
        batch_size=16,
        num_workers=0)

    for images, targets in data_loader:
        print(targets)