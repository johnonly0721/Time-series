__all__ = ['Transpose', 'get_activation_fn', 'moving_avg', 'series_decomp',
           'sin_cos_pos_encoding', 'Coord2dPosEncoding', 'Coord1dPosEncoding', 'positional_encoding']

import torch
import torch.nn as nn
import math


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False) -> None:
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class MovingAvg(nn.Module):
    """实现了一个移动平均层，用于突出时间序列的趋势。

    移动平均是一种常用的时间序列数据平滑技术，主要用于减少随机波动和突出长期趋势。
    本模块通过使用一维平均池化（AvgPool1d）来计算移动平均。

    参数:
        - kernel_size (int): 移动平均的窗口大小，决定了计算平均值时考虑多少个连续样本。
        - stride (int): 移动平均的步长，即窗口移动时的距离。

    输入:
        - x (Tensor): 输入的时间序列数据，期望的形状为 (B, L, C)，其中
                    B 是批次大小，L 是序列长度，C 是通道数。

    输出:
        - Tensor: 经过移动平均处理的时间序列数据，输出形状取决于kernel_size和stride。
    """

    def __init__(self, kernel_size, stride) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 对时间序列数据的两端进行填充，以便在边界处也能应用移动平均
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x


class SeriesDecomp(nn.Module):
    """
    时间序列分解模块，用于从原始时间序列中分离出趋势成分和残差成分。

    该模块使用移动平均块来计算时间序列的趋势成分，然后将趋势成分从原始序列中减去以得到残差成分。

    参数:
        - kernel_size (int): 移动平均窗口的大小。

    输入:
        - x (Tensor): 原始时间序列，期望的形状为 (B, L, C)，其中
                    B 是批次大小，L 是序列长度，C 是通道数。

    输出:
        - Tuple[Tensor, Tensor]: 包含两个元素的元组，第一个元素是残差成分，第二个元素是趋势成分。
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        ma = self.moving_avg(x)
        res = x - ma
        return res, ma


def sin_cos_pos_encoding(seq_len, d_model, norm=True):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if norm:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)


def coor2d_pos_encoding(seq_len, d_model, exp=False, norm=True, eps=1e-3, verbose=False):
    """
    生成二维坐标位置编码。

    该函数生成一个二维坐标网格的位置编码，可以用于模型中对位置信息的编码。

    参数:
        - seq_len (int): 序列长度，即编码的第一个维度大小。
        - d_model (int): 模型特征维度，即编码的第二个维度大小。
        - exp (bool): 是否使用指数加权的位置编码。
        - norm (bool): 是否对位置编码进行归一化处理。
        - eps (float): 归一化过程中允许的误差范围。
        - verbose (bool): 是否打印调试信息。

    输出:
        - Tensor: 位置编码，其形状为 (q_len, d_model)。
    """
    x = .5 if exp else 1
    i = 0
    for i in range(200):
        cpe = 2 * (torch.linspace(0, 1, seq_len).reshape(-1, 1) ** x) * \
            (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        if verbose:
            print(f'{i} iter: exp: {x}, cpe mean: {cpe.std()}')
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += .001
        else:
            x -= .001
        i += 1
    if norm:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def coor1d_pos_encoding(seq_len, exp=False, norm=True):
    """
    生成一维坐标位置编码。

    该函数生成一个一维坐标网格的位置编码，可以用于模型中对位置信息的编码。

    参数:
        - seq_len (int): 序列长度，即编码的第一个维度大小。
        - exp (bool): 是否使用指数加权的位置编码。
        - norm (bool): 是否对位置编码进行归一化处理。

    输出:
        - Tensor: 位置编码，其形状为 (seq_len, 1)。
    """
    cpe = 2 * (torch.linspace(0, 1, seq_len).reshape(-1, 1)
               ** (.5 if exp else 1)) - 1
    if norm:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe
