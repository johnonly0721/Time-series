import torch
import torch.nn as nn

from layers.Transformer.embeddings import SinCosPosEmbedding


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

# ==========位置编码==========

class Coor2dLinearPosEmbedding(nn.Module):
    def __init__(self, q_len, d_model, exp=False, norm=True, eps=1e-3) -> None:
        super().__init__()
        x = .5 if exp else 1
        i = 0
        for i in range(200):
            cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
            cpe.requires_grad = False
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
        self.cpe = cpe
        
    def forward(self, x):
        return self.cpe.unsqueeze(0).to(x.device).detach()

class Coor1dLinearPosEmbedding(nn.Module):
    def __init__(self, q_len, exp=False, norm=True) -> None:
        super().__init__()
        cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (.5 if exp else 1)) - 1) # (q_len, 1)
        if norm:
            cpe = cpe - cpe.mean()
            cpe = cpe / (cpe.std() * 10)
        cpe.requires_grad = False
        self.cpe = cpe
    
    def forward(self, x):
        return self.cpe.unsqueeze(0).unsqueeze(0).to(x.device).detach()

def get_pos_embedding(pe_type, learn_pe, q_len, d_model):
    if pe_type == None:
        W_pos = torch.empty((1, q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe_type == 'zero':
        W_pos = torch.empty((1, q_len, 1))
        nn.init.uniform_(W_pos, -0.02, -0.02)
    elif pe_type == 'normal' or pe_type == 'guass':
        W_pos = torch.zeros((1, q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe_type == 'uniform':
        W_pos = torch.zeros((1, q_len, 1))
        torch.nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe_type == 'lin1d':
        W_pos = Coor1dLinearPosEmbedding(q_len, exp=False, norm=True)