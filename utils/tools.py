import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import torch.nn as nn

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    """
    根据预设策略调整学习率。

    参数:
    - optimizer: PyTorch优化器。
    - scheduler: 学习率调度器。
    - epoch: 当前的训练轮次。
    - args: 包含学习率调整相关参数的对象。
    - printout: 是否打印学习率更新信息。

    功能描述:
    根据当前epoch和预设策略调整优化器的学习率。
    """
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.lr if epoch <
                     3 else args.lr * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.lr}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.lr if epoch <
                     10 else args.lr*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.lr if epoch <
                     15 else args.lr*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.lr if epoch <
                     25 else args.lr*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.lr if epoch <
                     5 else args.lr*0.1}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print(f'更新学习率为: {lr}')


class EarlyStopping:
    """
    提供早停机制以避免过拟合。

    参数:
    - patience: 无改进的epoch数，超过此值则停止训练。
    - verbose: 是否打印详细信息。
    - delta: 模型验证损失改进的最小变化量。

    方法:
    - __call__(val_loss, model, path): 根据验证损失判断是否应该早停。
    - save_checkpoint(val_loss, model, path): 保存当前最佳模型。
    """

    def __init__(self, patience=7, verbose=False, delta=0) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'早停计数器: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'验证损失减小 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型 ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class DotDict(dict):
    """
    点符号访问字典属性的字典类。

    功能描述:
    允许使用点号访问字典项，例如dict.key代替dict['key']。
    """
    __get_attr__ = dict.get
    __setattr__ = dict.__setitem__
    __del_attr__ = dict.__delitem__


class StandardScaler:
    """
    数据标准化处理类。

    参数:
    - mean: 数据的均值。
    - std: 数据的标准差。

    方法:
    - transform(data): 将数据标准化。
    - inverse_transform(data): 将标准化后的数据还原。
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    可视化真实值和预测值。

    参数:
    - true: 真实值序列。
    - preds: 预测值序列，可选。
    - name: 图片保存路径。

    功能描述:
    绘制并保存真实值和预测值的对比图。
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def test_params_flop(model, x_shape):
    """
    测试模型的参数数量和计算复杂度（FLOPs）。

    参数:
    - model: 要测试的模型。
    - x_shape: 输入数据的形状。

    功能描述:
    打印模型的参数数量和计算复杂度。
    """
    model_params = 0
    for param in model.parameters():
        model_params += param.numel()
        print('提示: 模型参数量为: %.2fM' % (model_params / 1e6))
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


class Timer:
    def __init__(self) -> None:
        self.start_time = None
        self.end_time = None
    
    def start(self):
        if self.start_time:
            print('Timer is running! Use .stop() to stop it.')
            return
        self.start_time = time.time()
        self.end_time = None
    
    def elapsed(self, reset_start=False):
        if self.start_time is None:
            print('Timer is not running! Use .start() to start it.')
            return
        st = self.start_time
        if reset_start:
            self.start_time = time.time()
        return time.time() - st
    
    def stop(self):
        if self.start_time is None:
            print('Timer is not running! Use .start() to start it.')
            return
        self.end_time = time.time()
        st = self.start_time
        self.start_time = None
        return self.end_time - st
    
def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'gelu':
        return nn.GELU()
    raise ValueError(f'{activation} 不可用。你可以使用 "relu", "gelu" 或者一个可调用的激活函数')
