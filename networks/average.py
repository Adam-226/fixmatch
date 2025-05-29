from copy import deepcopy

import torch


class ModelEMA(object):
    """
    指数移动平均(Exponential Moving Average)模型
    用于在训练过程中维护模型参数的移动平均，提高模型稳定性和性能
    """
    def __init__(self, args, model, decay):
        """
        初始化EMA模型
        Args:
            args: 包含设备信息的参数对象
            model: 原始训练模型
            decay: EMA衰减率，通常接近1.0（如0.999）
        """
        self.ema = deepcopy(model)  # 创建模型的深拷贝
        self.ema.to(args.device)    # 移动到指定设备
        self.ema.eval()             # 设置为评估模式
        self.decay = decay          # 衰减率
        self.ema_has_module = hasattr(self.ema, 'module')  # 检查是否为分布式模型
        
        # 修复EMA问题，感谢 https://github.com/valencebond/FixMatch_pytorch
        self.param_keys = [k for k, _ in self.ema.named_parameters()]  # 参数键列表
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]    # 缓冲区键列表
        
        # 禁用EMA模型的梯度计算
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """
        更新EMA模型参数
        使用公式: ema_param = decay * ema_param + (1 - decay) * model_param
        Args:
            model: 当前训练的模型
        """
        # 检查模型是否需要module前缀（分布式训练）
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        
        with torch.no_grad():
            # 获取模型状态字典
            msd = model.state_dict()    # 当前模型状态
            esd = self.ema.state_dict() # EMA模型状态
            
            # 更新模型参数
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k  # 分布式模型需要添加module前缀
                else:
                    j = k
                model_v = msd[j].detach()  # 当前模型参数值
                ema_v = esd[k]             # EMA模型参数值
                # 应用EMA更新公式
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            # 更新缓冲区（如BatchNorm的running_mean和running_var）
            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                # 缓冲区直接复制，不应用EMA
                esd[k].copy_(msd[j]) 