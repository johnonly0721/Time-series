import torch
import torch.nn as nn

import numpy as np
from math import sqrt

from utils.masking import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    """
    实现全注意力机制的类

    参数:
        mask_flag(bool): 是否使用掩码来避免未来信息的泄露
        factor(int): 用于概率注意力的因子, 这里不直接使用
        scale(float): 缩放因子, 通常是1/sqrt(d_k), 其中d_k是key的维度
        attention_dropout: 注意力权重的dropout率
        output_attention: 是否输出注意力权重

    方法:
        forward(queries, keys, values, attn_mask): 根据输入的查询、键、值和掩码计算注意力和输出
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        """根据输入的查询、键、值和掩码计算注意力和输出

        参数:
            queries (Tensor): 查询矩阵，形状为 (B, L, H, E)，其中 B 是批次大小，L 是查询序列长度，H 是头的数量，E 是每个头的特征维度
            keys (Tensor): 键矩阵，特征维度通常与查询矩阵相同，但序列长度可以不同，表示为 S，形状为 (B, S, H, E)
            values (Tensor): 值矩阵，形状为 (B, S, H, D)，其中 D 是值的特征维度
            attn_mask (Tensor): 注意力掩码，形状为 (B, L, S) 或其他与注意力分数矩阵兼容的形状，用于指定哪些位置不应参与注意力计算

        返回:
            输出 (Tensor): 注意力机制的输出，形状为 (B, L, H, D)
            注意力权重 (Tensor): 如果 self.output_attention 为 True，则返回注意力权重矩阵，形状为 (B, H, L, S)；否则返回 None

        计算过程描述:
            1. 计算查询和键的点积，得到原始的注意力分数。
            2. 如果提供了掩码，则应用掩码来修改分数，防止未来信息的泄露。
            3. 将分数乘以缩放因子，然后应用 softmax 函数得到注意力权重。
            4. 使用注意力权重对值进行加权求和，得到最终的输出。
            5. 根据 self.output_attention 返回相应的输出和注意力权重。
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum('blhe,bshe->bhls', queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
                scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum('bhls,bshd->blhd', A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    """
    概率注意力机制的类(Informer)

    参数:
        mask_flag(bool): 是否使用掩码来避免未来信息的泄露
        factor(int): 用于概率采样的因子，控制采样的稀疏程度
        scale(float): 缩放因子, 通常是1/sqrt(d_k), 其中d_k是key的维度
        attention_dropout: 注意力权重的dropout率
        output_attention: 是否输出注意力权重
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.factor = factor
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        概率采样并计算查询（Q）和键（K）之间的注意力分数的方法

        参数:
            Q (Tensor): 查询矩阵，形状为 (B, H, L_Q, E)，其中 B 是批次大小，H 是头的数量，L_Q 是查询序列长度，E 是特征维度
            K (Tensor): 键矩阵，形状为 (B, H, L_K, E)，其中 L_K 是键序列长度
            sample_k (int): 从键（K）中采样的点的数量
            n_top (int): 从采样的结果中选择最高的n_top个点进行计算

        返回:
            Q_K (Tensor): 近似的注意力分数矩阵，形状为 (B, H, n_top, L_K)
            M_top (Tensor): 选取的最高n_top个点的索引

        计算步骤:
            1. 对于每个q，从键（K）中采样 sample_k 个点，得到 K_sample
            2. 计算查询（Q）和采样的键（K_sample）之间的点积，得到 Q_K_sample
            3. 对每个q的点积分数进行稀疏性度量（最大值 - 平均值），挑选出最大的 n_top 个q索引，得到 M_top
            4. 从原始的Q中挑选出 M_top 对应的q，得到 Q_reduce
            5. 计算 Q_reduce 和 K 之间的点积，得到 Q_K
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # (L_Q, sample_k)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(
            1), index_sample, :]  # (B, H, L_Q, sample_k, E)
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()  # (B, H, L_Q, sample_k)

        M = Q_K_sample.max(-1)[0] - \
            torch.div(Q_K_sample.sum(-1), L_K)  # (B, H, L_Q)
        M_top = M.topk(n_top, sorted=False)[1]  # (B, H, n_top)

        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(
            H)[None, :, None], M_top, :]  # (B, H, n_top, E)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # (B, H, n_top, L_K)

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """

        初始化上下文，用于概率注意力机制中的初始状态
        Args:
            V (Tensor): 值矩阵，形状为 B, H, L_V, D，其中 B 是批次大小，H 是头的数量，L_V 是值序列长度，D 是特征维度
            L_Q (_type_): 查询序列的长度

        Returns:
            context (Tensor): 初始化的上下文，形状为 (B, H, L_Q, D) 或 (B, H, L_V, D)，取决于是否使用掩码

        计算过程:
            1. 如果没有使用掩码（self.mask_flag 为 False），则计算值（V）的平均值作为每个头的初始上下文
            2. 如果使用掩码（self.mask_flag 为 True），则计算值（V）的累积和作为初始上下文，确保每个位置只能访问到它之前的位置的信息
            3. 返回计算得到的初始上下文，它将被用于后续的注意力更新
        """
        B, H, L_V, D = V.shape
        if self.mask_flag:
            # 使用掩码
            assert (L_Q == L_V)
            context = V.cumsum(dim=-2)
        else:
            # 不使用掩码
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H,
                                                 L_Q, V_sum.shape[-1]).clone()
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """更新上下文信息，根据当前的注意力分数和值矩阵

        参数:
            context_in (Tensor): 输入的上下文，形状为 (B, H, L_Q, D)
            V (Tensor): 值矩阵，形状为 (B, H, L_V, D)
            scores (Tensor): 注意力分数，形状为 (B, H, L_Q, L_K) 或其他兼容形状
            index (Tensor): 选定的最重要的键的索引
            L_Q (int): 查询序列的长度
            L_Q (int): 查询序列的长度

        返回:
            更新后的上下文 (Tensor)
            注意力权重 (Tensor): 如果 self.output_attention 为 True，则返回；否则为 None

        计算过程:
            1. 如果使用掩码，应用掩码到注意力分数上
            2. 计算注意力权重
            3. 使用注意力权重和值矩阵更新输入的上下文
            4. 如果需要，处理并返回注意力权重
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # (B, H, L_Q, L_K)
        context_in[torch.arange(B)[:, None, None], torch.arange(
            H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            attns = torch.ones([B, H, L_V, L_V] /
                               L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)  # (B, H, L_Q, D) -> (B, H, L_Q, D)
        keys = keys.transpose(2, 1)  # (B, H, L_K, D) -> (B, H, L_K, D)
        values = values.transpose(2, 1)  # (B, H, L_V, D) -> (B, H, L_V, D)

        U_part = self.factor * \
            np.ceil(np.log(L_K)).astype('int').item()  # c * ln(L_K)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c * ln(L_Q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        # (B, H, n_top, L_K), (B, H, n_top)
        scores_top, index = self._prob_QK(queries, keys, U_part, u)

        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)  # (B, H, L_Q, D)
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)  # (B, H, L_Q, D), (B, H, L_Q, L_K)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    """封装了自注意力机制的神经网络层

    参数:
    - attention: 指定的注意力机制实例（FullAttention 或 ProbAttention）。
    - d_model: 输入和输出的特征维度。
    - n_heads: 注意力头的数量，即将 d_model 分割成多少个头。
    - d_keys (可选): 每个头的键的维度，默认为 d_model // n_heads。
    - d_values (可选): 每个头的值的维度，默认为 d_model // n_heads。

    方法:
    - forward(queries, keys, values, attn_mask): 根据输入的查询、键、值和注意力掩码计算并返回注意力机制的输出。
    """

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _,  = queries.shape
        _, S, _,  = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
