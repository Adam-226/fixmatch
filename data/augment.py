# 本文件中的代码改编自以下项目:
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10  # 参数最大值


def AutoContrast(img, **kwarg):
    """
    自动对比度调整
    Args:
        img: PIL图像
    Returns:
        调整对比度后的图像
    """
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    """
    亮度调整
    Args:
        img: PIL图像
        v: 亮度调整值
        max_v: 最大调整值
        bias: 偏置值
    Returns:
        调整亮度后的图像
    """
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    """
    颜色饱和度调整
    Args:
        img: PIL图像
        v: 饱和度调整值
        max_v: 最大调整值
        bias: 偏置值
    Returns:
        调整饱和度后的图像
    """
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    """
    对比度调整
    Args:
        img: PIL图像
        v: 对比度调整值
        max_v: 最大调整值
        bias: 偏置值
    Returns:
        调整对比度后的图像
    """
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    """
    Cutout数据增强：在图像上随机遮挡矩形区域
    Args:
        img: PIL图像
        v: 遮挡大小参数
        max_v: 最大遮挡大小
        bias: 偏置值
    Returns:
        应用Cutout后的图像
    """
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))  # 计算遮挡区域大小
    return CutoutAbs(img, v)


def CutoutAbs(img, v):
    """
    绝对大小的Cutout
    Args:
        img: PIL图像
        v: 遮挡区域的边长
    Returns:
        应用Cutout后的图像
    """
    w, h = img.size
    x0 = np.random.uniform(0, w)  # 随机选择遮挡区域的中心点
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))  # 计算遮挡区域的左上角
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))       # 计算遮挡区域的右下角
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # 用灰色填充遮挡区域
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    """
    直方图均衡化
    Args:
        img: PIL图像
    Returns:
        均衡化后的图像
    """
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    """
    图像反转（负片效果）
    Args:
        img: PIL图像
    Returns:
        反转后的图像
    """
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    """
    色调分离：减少图像的颜色位数
    Args:
        img: PIL图像
        v: 分离程度
        max_v: 最大分离程度
        bias: 偏置值
    Returns:
        色调分离后的图像
    """
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    """
    图像旋转
    Args:
        img: PIL图像
        v: 旋转角度参数
        max_v: 最大旋转角度
        bias: 偏置值
    Returns:
        旋转后的图像
    """
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v  # 随机决定旋转方向
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    """
    锐度调整
    Args:
        img: PIL图像
        v: 锐度调整值
        max_v: 最大调整值
        bias: 偏置值
    Returns:
        调整锐度后的图像
    """
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    """
    X轴剪切变换
    Args:
        img: PIL图像
        v: 剪切程度
        max_v: 最大剪切程度
        bias: 偏置值
    Returns:
        剪切变换后的图像
    """
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    """
    Y轴剪切变换
    Args:
        img: PIL图像
        v: 剪切程度
        max_v: 最大剪切程度
        bias: 偏置值
    Returns:
        剪切变换后的图像
    """
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    """
    曝光过度效果：反转超过阈值的像素
    Args:
        img: PIL图像
        v: 阈值参数
        max_v: 最大阈值
        bias: 偏置值
    Returns:
        曝光过度效果后的图像
    """
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    """
    增强曝光过度效果
    Args:
        img: PIL图像
        v: 增强程度
        max_v: 最大增强程度
        bias: 偏置值
        threshold: 阈值
    Returns:
        增强曝光过度效果后的图像
    """
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    """
    X轴平移
    Args:
        img: PIL图像
        v: 平移距离参数
        max_v: 最大平移距离
        bias: 偏置值
    Returns:
        平移后的图像
    """
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])  # 转换为像素距离
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    """
    Y轴平移
    Args:
        img: PIL图像
        v: 平移距离参数
        max_v: 最大平移距离
        bias: 偏置值
    Returns:
        平移后的图像
    """
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])  # 转换为像素距离
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    """
    将整数参数转换为浮点数参数
    Args:
        v: 输入参数值
        max_v: 最大参数值
    Returns:
        归一化的浮点数参数
    """
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    """
    将参数转换为整数参数
    Args:
        v: 输入参数值
        max_v: 最大参数值
    Returns:
        归一化的整数参数
    """
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs


def my_augment_pool():
    """
    定义数据增强操作池
    Returns:
        增强操作列表，每个元素为(函数, 最大值, 偏置值)的元组
    """
    # 测试用的增强操作池
    augs = [(AutoContrast, None, None),      # 自动对比度
            (Brightness, 1.8, 0.1),         # 亮度调整
            (Color, 1.8, 0.1),              # 颜色饱和度
            (Contrast, 1.8, 0.1),           # 对比度
            (Cutout, 0.2, 0),               # Cutout遮挡
            (Equalize, None, None),          # 直方图均衡化
            (Invert, None, None),            # 图像反转
            (Posterize, 4, 4),              # 色调分离
            (Rotate, 30, 0),                # 旋转
            (Sharpness, 1.8, 0.1),          # 锐度
            (ShearX, 0.3, 0),               # X轴剪切
            (ShearY, 0.3, 0),               # Y轴剪切
            (Solarize, 256, 0),             # 曝光过度
            (SolarizeAdd, 110, 0),          # 增强曝光过度
            (TranslateX, 0.45, 0),          # X轴平移
            (TranslateY, 0.45, 0)]          # Y轴平移
    return augs


class RandAugmentPC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = my_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img


class RandAugmentMC(object):
    """
    RandAugment数据增强类（多类别版本）
    论文: RandAugment: Practical automated data augmentation with a reduced search space
    """
    def __init__(self, n, m):
        """
        Args:
            n: 每次随机选择的增强操作数量
            m: 增强操作的强度（0-10）
        """
        self.n = n  # 操作数量
        self.m = m  # 操作强度
        self.augment_pool = my_augment_pool()  # 增强操作池

    def __call__(self, img):
        """
        对图像应用RandAugment
        Args:
            img: PIL图像
        Returns:
            增强后的图像
        """
        ops = random.choices(self.augment_pool, k=self.n)  # 随机选择n个操作
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)  # 随机选择操作强度
            if random.random() < 0.5:  # 50%概率应用该操作
                img = op(img, v=v, max_v=max_v, bias=bias)
        return img 