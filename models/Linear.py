import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: (batch_size, seq_len, channel)
        x = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x  # (batch_size, pred_len, channel)
