import torch

class TriangularCausalMask():
    """
    Mask for Transformer encoder.
    B: batch size
    L: sequence length
    """
    def __init__(self, B, L, device='cpu'):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
    
    @property
    def mask(self):
        return self._mask

class ProbMask():
    """
    ProbMask is used to mask the attention scores.
    B: batch size
    H: number of heads
    L: sequence length
    index: index of the current token
    scores: attention scores, shape [B, H, 1, L]
    """
    def __init__(self, B, H, L, index, scores, device='cpu'):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1) # Upper triangular part of a matrix(2-D tensor) is zero
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1]) # Expand the mask to the same shape as scores
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                                     torch.arange(H)[None, :, None], 
                                     index, :].to(device) #
        print(indicator.shape)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask