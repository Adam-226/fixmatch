import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def mish(x):
    """
    Mish激活函数: A Self Regularized Non-Monotonic Neural Activation Function
    论文: https://arxiv.org/abs/1908.08681
    Args:
        x: 输入张量
    Returns:
        应用Mish激活后的张量
    """
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """
    如何在半监督学习中正确使用BatchNorm
    论文: https://arxiv.org/abs/2001.11216
    """

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class ResNeXtBottleneck(nn.Module):
    """
    ResNeXt瓶颈块 (类型C)
    论文: Aggregated Residual Transformations for Deep Neural Networks
    参考: https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua
    """

    def __init__(self, in_channels, out_channels, stride,
                 cardinality, base_width, widen_factor):
        """
        构造函数
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 卷积步长，替代池化层
            cardinality: 卷积组数（ResNeXt的核心参数）
            base_width: 每组的基础通道数
            widen_factor: 用于在卷积前减少输入维度的因子
        """
        super().__init__()
        # 计算中间层的通道数
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        
        # 1x1卷积：降维
        self.conv_reduce = nn.Conv2d(
            in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D, momentum=0.001)
        
        # 3x3分组卷积：ResNeXt的核心创新
        self.conv_conv = nn.Conv2d(D, D,
                                   kernel_size=3, stride=stride, padding=1,
                                   groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D, momentum=0.001)
        self.act = mish  # 激活函数
        
        # 1x1卷积：升维
        self.conv_expand = nn.Conv2d(
            D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels, momentum=0.001)

        # 残差连接的快捷路径
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            # 当输入输出通道数不同时，需要调整维度
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels,
                                               kernel_size=1,
                                               stride=stride,
                                               padding=0,
                                               bias=False))
            self.shortcut.add_module(
                'shortcut_bn', nn.BatchNorm2d(out_channels, momentum=0.001))

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征图
        Returns:
            输出特征图
        """
        # 瓶颈结构：1x1降维 -> 3x3分组卷积 -> 1x1升维
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = self.act(self.bn_reduce.forward(bottleneck))
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = self.act(self.bn.forward(bottleneck))
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        
        # 残差连接
        residual = self.shortcut.forward(x)
        return self.act(residual + bottleneck)


class CifarResNeXt(nn.Module):
    """
    针对CIFAR数据集优化的ResNeXt网络
    论文: https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, cardinality, depth, num_classes,
                 base_width, widen_factor=4):
        """
        构造函数
        Args:
            cardinality: 卷积组数（ResNeXt的关键超参数）
            depth: 网络深度（层数）
            num_classes: 分类类别数
            base_width: 每组的基础通道数
            widen_factor: 通道维度调整因子
        """
        super().__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9  # 每个阶段的块数
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = num_classes
        self.output_size = 64
        
        # 各阶段的通道数配置
        self.stages = [64, 64 * self.widen_factor, 128 *
                       self.widen_factor, 256 * self.widen_factor]

        # 初始卷积层
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64, momentum=0.001)
        self.act = mish
        
        # 三个主要阶段
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        
        # 分类器
        self.classifier = nn.Linear(self.stages[3], num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """
        构建包含n个瓶颈模块的块，其中n从网络深度推断
        Args:
            name: 当前块的字符串名称
            in_channels: 输入通道数
            out_channels: 输出通道数
            pool_stride: 在块的第一个瓶颈中减少空间维度的因子
        Returns:
            由n个连续瓶颈块组成的模块
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                # 第一个瓶颈块：可能改变空间维度和通道数
                block.add_module(name_, ResNeXtBottleneck(in_channels,
                                                          out_channels,
                                                          pool_stride,
                                                          self.cardinality,
                                                          self.base_width,
                                                          self.widen_factor))
            else:
                # 后续瓶颈块：保持维度不变
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels,
                                                   out_channels,
                                                   1,
                                                   self.cardinality,
                                                   self.base_width,
                                                   self.widen_factor))
        return block

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像张量 [batch_size, 3, 32, 32]
        Returns:
            分类logits [batch_size, num_classes]
        """
        # 初始卷积和激活
        x = self.conv_1_3x3.forward(x)
        x = self.act(self.bn_1.forward(x))
        
        # 三个主要阶段
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        
        # 全局平均池化和分类
        x = F.adaptive_avg_pool2d(x, 1)  # 自适应平均池化到1x1
        x = x.view(-1, self.stages[3])   # 展平
        return self.classifier(x)


def build_resnext(cardinality, depth, width, num_classes):
    """
    构建ResNeXt模型
    Args:
        cardinality: 卷积组数
        depth: 网络深度
        width: 基础宽度
        num_classes: 分类类别数
    Returns:
        ResNeXt模型
    """
    logger.info(f"Model: ResNeXt {depth+1}x{width}")
    return CifarResNeXt(cardinality=cardinality,
                        depth=depth,
                        base_width=width,
                        num_classes=num_classes) 