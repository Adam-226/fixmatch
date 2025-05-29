import logging
import math

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
    论文: https://arxiv.org/abs/2006.10740
    """

    def __init__(self, num_features, alpha=0.99, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x)


class BasicBlock(nn.Module):
    """
    WideResNet的基本残差块
    """
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        """
        Args:
            in_planes: 输入通道数
            out_planes: 输出通道数
            stride: 卷积步长
            dropRate: Dropout概率
            activate_before_residual: 是否在残差连接前激活
        """
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                 padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    """
    WideResNet的网络块，包含多个BasicBlock
    """
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        """
        Args:
            nb_layers: 块中的层数
            in_planes: 输入通道数
            out_planes: 输出通道数
            block: 基本块类型
            stride: 步长
            dropRate: Dropout概率
            activate_before_residual: 是否在残差连接前激活
        """
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        """
        构建网络层
        """
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """
    WideResNet网络架构
    论文: Wide Residual Networks (https://arxiv.org/abs/1605.07146)
    """
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0):
        """
        Args:
            num_classes: 分类类别数
            depth: 网络深度
            widen_factor: 宽度因子
            dropRate: Dropout概率
        """
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6  # 每个块的层数
        block = BasicBlock
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        # 三个主要的网络块
        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], block, 2, dropRate)
        
        # 最后的BatchNorm和分类器
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像张量 [batch_size, 3, 32, 32]
        Returns:
            分类logits [batch_size, num_classes]
        """
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)  # 全局平均池化
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def build_wideresnet(depth, widen_factor, dropout, num_classes):
    """
    构建WideResNet模型
    Args:
        depth: 网络深度
        widen_factor: 宽度因子
        dropout: Dropout概率
        num_classes: 分类类别数
    Returns:
        WideResNet模型
    """
    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    return WideResNet(num_classes, depth, widen_factor, dropout) 