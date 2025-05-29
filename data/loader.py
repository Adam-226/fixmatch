import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .augment import RandAugmentMC

logger = logging.getLogger(__name__)

# CIFAR-10数据集的均值和标准差（用于归一化）
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
# CIFAR-100数据集的均值和标准差
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
# 标准归一化参数
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root):
    """
    获取CIFAR-10数据集的标记、未标记和测试数据
    Args:
        args: 包含数据集配置的参数对象
        root: 数据集根目录
    Returns:
        标记数据集、未标记数据集、测试数据集
    """
    # 标记数据的变换（弱增强）
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(size=32,      # 随机裁剪
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),              # 转换为张量
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)  # 归一化
    ])
    
    # 测试数据的变换（无增强）
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    
    # 加载基础CIFAR-10数据集
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    # 划分标记和未标记数据的索引
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    # 创建标记数据集
    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    # 创建未标记数据集（使用FixMatch变换：弱增强+强增强）
    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    # 创建测试数据集
    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):
    """
    获取CIFAR-100数据集的标记、未标记和测试数据
    Args:
        args: 包含数据集配置的参数对象
        root: 数据集根目录
    Returns:
        标记数据集、未标记数据集、测试数据集
    """
    # 标记数据的变换
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    # 测试数据的变换
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    # 加载基础CIFAR-100数据集
    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    # 划分标记和未标记数据的索引
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    # 创建标记数据集
    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    # 创建未标记数据集
    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    # 创建测试数据集
    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    """
    将数据划分为标记和未标记两部分
    Args:
        args: 包含标记数据数量等配置的参数对象
        labels: 数据标签列表
    Returns:
        标记数据索引、未标记数据索引
    """
    # 每个类别的标记样本数量
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    
    # 未标记数据：使用所有数据（包括标记数据）
    # 参考：https://github.com/kekmodel/FixMatch-pytorch/issues/10
    unlabeled_idx = np.array(range(len(labels)))
    
    # 为每个类别随机选择指定数量的标记样本
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]  # 找到类别i的所有样本索引
        idx = np.random.choice(idx, label_per_class, False)  # 随机选择
        labeled_idx.extend(idx)
    
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    # 如果需要扩展标记数据以适应批次大小
    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        # 重复标记数据索引
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    
    np.random.shuffle(labeled_idx)  # 打乱顺序
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    """
    FixMatch的数据变换类
    对每个未标记样本生成弱增强和强增强两个版本
    """
    def __init__(self, mean, std):
        """
        Args:
            mean: 归一化均值
            std: 归一化标准差
        """
        # 弱增强：基本的数据增强
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        
        # 强增强：弱增强 + RandAugment
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])  # RandAugment强增强
        
        # 归一化变换
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        """
        对输入图像应用弱增强和强增强
        Args:
            x: 输入PIL图像
        Returns:
            弱增强图像、强增强图像（都已归一化）
        """
        weak = self.weak(x)      # 弱增强
        strong = self.strong(x)  # 强增强
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    """
    CIFAR-10半监督学习数据集类
    继承自torchvision的CIFAR10，支持索引子集
    """
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        """
        Args:
            root: 数据集根目录
            indexs: 要使用的数据索引
            train: 是否为训练集
            transform: 数据变换
            target_transform: 标签变换
            download: 是否下载数据集
        """
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        # 根据索引筛选数据
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        """
        获取指定索引的数据项
        Args:
            index: 数据索引
        Returns:
            图像、标签
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)  # 转换为PIL图像

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    """
    CIFAR-100半监督学习数据集类
    继承自torchvision的CIFAR100，支持索引子集
    """
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        """
        Args:
            root: 数据集根目录
            indexs: 要使用的数据索引
            train: 是否为训练集
            transform: 数据变换
            target_transform: 标签变换
            download: 是否下载数据集
        """
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        # 根据索引筛选数据
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        """
        获取指定索引的数据项
        Args:
            index: 数据索引
        Returns:
            图像、标签
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)  # 转换为PIL图像

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


# 数据集获取器字典
DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100} 