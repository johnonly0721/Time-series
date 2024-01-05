import torch
import torch.nn as nn
import math


class SinCosPosEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, norm=False):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()  # (max_len, d_model)
        pe.requires_grad = False

        position = torch.arange(
            0, max_len).float().unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-math.log(10000.0) / d_model))  # (d_model / 2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        if norm:
            pe = pe - pe.mean()
            pe = pe / (pe.std() * 10)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.pe = pe

    def forward(self, x):
        return self.pe[:, :x.size(1)]  # (1, seq_len, d_model)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model) -> None:
        super().__init__()

        sin_cos_pos_encoding = SinCosPosEmbedding(d_model)
        w = sin_cos_pos_encoding.pe.squeeze()

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embedding_type='fixed', freq='h') -> None:
        super().__init__()

        minute_size = 4
        hour_size = 4
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embedding = FixedEmbedding if embedding_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embedding = Embedding(minute_size, d_model)
        self.hour_embedding = Embedding(hour_size, d_model)
        self.weekday_embedding = Embedding(weekday_size, d_model)
        self.day_embedding = Embedding(day_size, d_model)
        self.month_embedding = Embedding(month_size, d_model)

    def forward(self, x):
        x = x.long()  # (batch_size, seq_len, 5) 或 (batch_size, seq_len, 4)

        minute_x = self.minute_embedding(x[:, :, 4]) if hasattr(
            self, 'minute_embedding') else 0.
        hour_x = self.hour_embedding(x[:, :, 3])
        weekday_x = self.weekday_embedding(x[:, :, 2])
        day_x = self.day_embedding(x[:, :, 1])
        # (batch_size, seq_len, d_model)
        month_x = self.month_embedding(x[:, :, 0])

        return minute_x + hour_x + weekday_x + day_x + month_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embedding_type='timeF', freq='h') -> None:
        super().__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_input = freq_map[freq]
        self.embedding = nn.Linear(d_input, d_model, bias=False)

    def forward(self, x):
        return self.embedding(x)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model) -> None:
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding,
                                   padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embedding_type='fixed', freq='h', dropout=0.1) -> None:
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = SinCosPosEmbedding(d_model)
        if embedding_type == 'timeF':
            self.temporal_embedding = TimeFeatureEmbedding(
                d_model, embedding_type, freq)
        else:
            self.temporal_embedding = TemporalEmbedding(
                d_model, embedding_type, freq)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        ve = self.value_embedding(x)
        te = self.temporal_embedding(x_mark)
        pe = self.position_embedding(x).to(x.device).detach()
        return self.dropout(x)


class DataEmbeddingNoPosition(nn.Module):
    def __init__(self, c_in, d_model, embedding_type='fixed', freq='h', dropout=0.1) -> None:
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in, d_model)
        if embedding_type == 'timeF':
            self.temporal_embedding = TimeFeatureEmbedding(
                d_model, embedding_type, freq)
        else:
            self.temporal_embedding = TemporalEmbedding(
                d_model, embedding_type, freq)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbeddingNoTemporal(nn.Module):
    def __init__(self, c_in, d_model, embedding_type='fixed', freq='h', dropout=0.1) -> None:
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = SinCosPosEmbedding()(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x).to(x.device).detach()
        return self.dropout(x)


class DataEmbeddingNoPosAndTemp(nn.Module):
    def __init__(self, c_in, d_model, embedding_type='fixed', freq='h', dropout=0.1) -> None:
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)
