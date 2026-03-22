
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer():
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else lambda x: x
        self.merged = False
        self.merge_weights = merge_weights

class Linear(nn.Linear, LoRALayer):
    def __init__(self, in_features: int, out_features: int, r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0., **kwargs):

        illegal_keys = ['information', 'mix_weight', 'downsample', 'rank_list', 'fan_in_fan_out', 'merge_weights']
        for key in illegal_keys:
            kwargs.pop(key, None)
            
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=True)

        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.weight, bias=self.bias)
        if self.r > 0:
            result += (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return result

class Conv2d(nn.Conv2d, LoRALayer):
    def __init__(self, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., **kwargs):

        illegal_keys = ['information', 'mix_weight', 'downsample', 'rank_list', 'merge_weights']
        for key in illegal_keys:
            kwargs.pop(key, None)
            
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=True)
        
    def forward(self, x):
        return super().forward(x)

def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False