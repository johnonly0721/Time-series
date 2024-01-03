import os
import torch


class Exp_Basic:
    """
    基础实验类，提供了构建、训练和测试模型的基本框架
    
    方法:
        __init__(self, args): 构造函数，初始化实验设置和模型参数
        _build_model: 构建模型的方法，需要在子类中实现。
        _acquire_device: 确定并设置计算设备（CPU或GPU）
        _get_data: 获取数据的方法，通常在子类中实现具体的数据加载逻辑
        val: 验证方法，用于在验证集上评估模型性能
        test: 测试方法，用于在测试集上评估模型性能
        train: 训练方法，用于训练模型
    
    参数:
        args: 包含所有实验设置和模型参数的对象
    """
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Using GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('Using CPU')
        return device

    def _get_data(self):
        pass

    def validate(self):
        pass

    def test(self):
        pass

    def train(self):
        pass
    
    def infer(self):
        pass
