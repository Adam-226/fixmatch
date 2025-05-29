# 半监督图像分类实验 - FixMatch实现

本项目是基于PyTorch实现的FixMatch半监督学习算法，用于CIFAR-10数据集的图像分类任务。

## 实验目的

### 关于半监督学习
神经网络模型通常需要大量标记好的训练数据来训练模型。然而，在许多情况下，获取大量标记好的数据可能是困难、耗时或昂贵的。这就是半监督学习的应用场景。半监督学习的核心思想是利用无标记数据的信息来改进模型的学习效果。在半监督学习中，我们使用少量标记数据进行有监督学习，同时利用大量无标记数据的信息。通过充分利用无标记数据的潜在结构和分布特征，半监督学习可以帮助模型更好地泛化和适应未标记数据。

### 半监督图像分类
半监督学习在图像分类取得了非常大的进步，涌现了许多经典的半监督图像分类算法，如：πModel、Mean Teacher、MixMatch、FixMatch等。这些算法都取得了非常好的结果，能够在仅使用少量标注数据的情况下，实现高精度的图像分类，在ImageNet、CIFAR-10、CIFAR-100等数据集上都有非常不错的效果。

## FixMatch算法原理

FixMatch结合了伪标签（Pseudo Label）和一致性正则化（Consistency Regularization）来实现对无标注数据的高效利用，训练过程包括两个部分：

1. **有监督训练**：有label的数据，执行有监督训练，和普通分类任务训练没有区别
2. **无监督训练**：没有label的数据，首先经过弱增强获取伪标签，然后利用该伪标签去监督强增强的输出值，只有大于一定阈值条件才执行伪标签的生成，并使用伪标签来进行无标注图像的训练

## 项目结构

```
.
├── configs/                # 配置文件目录
│   ├── cifar10_40.yaml     # CIFAR-10 40张标注数据配置
│   ├── cifar10_250.yaml    # CIFAR-10 250张标注数据配置
│   ├── cifar10_4000.yaml   # CIFAR-10 4000张标注数据配置
│   └── cifar100_10000.yaml # CIFAR-100 10000张标注数据配置
├── data/                   # 数据处理模块
│   ├── augment.py          # 数据增强（RandAugment）
│   └── loader.py           # CIFAR数据加载器
├── helpers/                # 工具函数
│   ├── __init__.py
│   └── tools.py            # 辅助工具函数
├── networks/               # 网络模型
│   ├── average.py          # 指数移动平均（EMA）
│   ├── backbone.py         # WideResNet网络
│   └── resnext.py          # ResNeXt网络
├── main.py                 # 主训练文件
├── requirements.txt        # 项目依赖包
└── README.md
```

## 实验要求

1. **算法实现**：基于PyTorch实现FixMatch半监督图像分类算法
2. **数据集实验**：在CIFAR-10数据集上进行实验，测试在40、250、4000张标注数据情况下的分类效果
3. **网络架构**：使用WideResNet-28-2作为Backbone网络（深度28，扩展因子2）
4. **对比实验**：与其他半监督算法（如MixMatch）进行对比分析

## 使用方法

### 环境要求
- Python 3.6+
- PyTorch 1.4+
- torchvision 0.5+
- tensorboard
- numpy
- tqdm
- PyYAML
- apex (可选，用于混合精度训练)

### 环境安装
```bash
# 安装所有依赖包
pip install -r requirements.txt

# 可选：安装apex用于混合精度训练（需要NVIDIA GPU）
# git clone https://github.com/NVIDIA/apex
# cd apex
# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### 训练命令

```bash
# CIFAR-10 40张标注数据
python main.py --config configs/cifar10_40.yaml

# CIFAR-10 250张标注数据
python main.py --config configs/cifar10_250.yaml

# CIFAR-10 4000张标注数据
python main.py --config configs/cifar10_4000.yaml

# CIFAR-100 10000张标注数据（分布式训练）
python -m torch.distributed.launch --nproc_per_node 4 ./main.py --config configs/cifar100_10000.yaml
```

**自动检查点恢复功能**：
- 训练会自动从上次中断的地方继续，无需手动指定检查点文件
- 检查点文件保存在输出目录下的 `checkpoint.pth.tar`
- 如果检查点存在，程序会自动加载并继续训练
- 支持中断后重新运行相同命令即可继续训练

### 监控训练进度
```bash
tensorboard --logdir=<your_output_directory>
```

### 主要参数说明

- `--config`: 配置文件路径
- `--dataset`: 数据集选择（cifar10/cifar100）
- `--num-labeled`: 标注数据数量
- `--arch`: 网络架构（wideresnet/resnext）
- `--batch-size`: 批次大小
- `--lr`: 学习率
- `--threshold`: 伪标签置信度阈值（默认0.95）
- `--lambda-u`: 无监督损失权重（默认1.0）
- `--mu`: 未标记数据批次大小倍数（默认7）
- `--T`: 伪标签温度参数（默认1.0）
- `--ema-decay`: EMA衰减率（默认0.999）

## 实验结果

### CIFAR-10数据集结果
| 标注数据量 | 40 | 250 | 4000 |
|:---:|:---:|:---:|:---:|
| 论文结果 | 86.19 ± 3.37 | 94.93 ± 0.65 | 95.74 ± 0.05 |
| 本实现 | 89.88 (173 epoch) | 89.42 (40 epoch) | 92.48 (40 epoch) |

## 参考文献

1. [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)
2. [A Survey on Deep Semi-supervised Learning](https://arxiv.org/abs/2103.00550)
3. [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)
4. [Mean Teachers are Better Role Models](https://arxiv.org/abs/1703.01780)

## 相关项目

- [Official TensorFlow implementation](https://github.com/google-research/fixmatch)
- [Microsoft Semi-supervised Learning](https://github.com/microsoft/Semi-supervised-learning)
- [TorchSSL](https://github.com/StephenStorm/TorchSSL)
