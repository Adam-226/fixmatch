'''
PyTorch辅助函数集合，包括：
    - get_mean_and_std: 计算数据集的均值和标准差
    - accuracy: 计算分类准确率
    - AverageMeter: 平均值计算器
'''
import logging

import torch

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter']


def get_mean_and_std(dataset):
    '''
    计算数据集的均值和标准差
    Args:
        dataset: PyTorch数据集对象
    Returns:
        mean: 各通道的均值张量
        std: 各通道的标准差张量
    '''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)  # RGB三个通道的均值
    std = torch.zeros(3)   # RGB三个通道的标准差
    logger.info('==> Computing mean and std..')
    
    # 遍历数据集计算统计量
    for inputs, targets in dataloader:
        for i in range(3):  # 对每个颜色通道
            mean[i] += inputs[:, i, :, :].mean()  # 累加均值
            std[i] += inputs[:, i, :, :].std()    # 累加标准差
    
    # 计算平均值
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """
    计算指定k值的分类准确率
    Args:
        output: 模型输出logits [batch_size, num_classes]
        target: 真实标签 [batch_size]
        topk: 要计算的top-k准确率元组，如(1, 5)
    Returns:
        res: 各个k值对应的准确率列表
    """
    maxk = max(topk)  # 最大的k值
    batch_size = target.size(0)

    # 获取top-k预测结果
    _, pred = output.topk(maxk, 1, True, True)  # [batch_size, maxk]
    pred = pred.t()  # 转置为 [maxk, batch_size]
    
    # 检查预测是否正确
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    # 计算各个k值的准确率
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)  # 前k个预测中正确的数量
        res.append(correct_k.mul_(100.0 / batch_size))      # 转换为百分比
    return res


class AverageMeter(object):
    """
    计算和存储平均值和当前值的工具类
    从 https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262 导入
    """

    def __init__(self):
        """初始化平均值计算器"""
        self.reset()

    def reset(self):
        """重置所有统计量"""
        self.val = 0    # 当前值
        self.avg = 0    # 平均值
        self.sum = 0    # 累计和
        self.count = 0  # 计数

    def update(self, val, n=1):
        """
        更新统计量
        Args:
            val: 新的数值
            n: 该数值的权重（默认为1）
        """
        self.val = val              # 更新当前值
        self.sum += val * n         # 累加到总和
        self.count += n             # 增加计数
        self.avg = self.sum / self.count  # 重新计算平均值 