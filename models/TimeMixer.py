import torch
from torch import nn
from layers.Patch_TST.PatchTST_layers import series_decomp

class DFT_series_decomp(nn.Module):
    
    def __init__(self, top_k=5) -> None:
        super().__init__()
        self.top_k = top_k
    
    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend

class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, seq_len:int, down_sampling_window:int, down_sampling_layers:int) -> None:
        super().__init__()
        self.down_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_sampling_window ** i), 
                        seq_len // (down_sampling_window ** (i+1))
                    ),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // (down_sampling_window ** (i+1)),
                        seq_len // (down_sampling_window ** (i+1))
                    )
                )
                for i in range(down_sampling_layers)
            ]
        )
    
    def forward(self, season_list:list):
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]
        
        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))
        return out_season_list

class MultiScaleTrendMixing(nn.Module):
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers) -> None:
        super().__init__()
        
        self.up_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_sampling_window ** (i+1)),
                        seq_len // (down_sampling_window ** i)
                    ),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** i)
                    )
                )
                for i in reversed(range(down_sampling_layers))
            ]
        )
    
    def forward(self, trend_list:list):
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]
        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))
        
        out_trend_list.reverse()
        return out_trend_list
    
class PastDecompMixing(nn.Module):
    def __init__(self, seq_len:int, pred_len:int, down_sampling_window:int, 
                 down_sampling_layers:int, d_model:int, d_ff:int,
                 dropout:float, channel_independence:bool, decomp_method:str,
                 moving_avg:int, top_k:int) -> None:
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.channel_independence = channel_independence
        
        if decomp_method == 'moving_avg':
            self.decomp = series_decomp(moving_avg)
        elif decomp_method == 'dft_decomp':
            self.decomp = DFT_series_decomp(top_k)
        else:
            raise ValueError('Decomposition method not supported')
        
        if not self.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_ff),
                nn.GELU(),
                nn.Linear(in_features=d_ff, out_features=d_model)
            )
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(seq_len=self.seq_len, 
                                                                down_sampling_window=self.down_sampling_window, 
                                                                down_sampling_layers=self.down_sampling_layers)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(seq_len=self.seq_len, 
                                                              down_sampling_window=self.down_sampling_window, 
                                                              down_sampling_layers=self.down_sampling_layers)
        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model)
        )
    
    def forward(self, x_list:list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)
        
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decomp(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season)
            trend_list.append(trend)
        
        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)
        
        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list