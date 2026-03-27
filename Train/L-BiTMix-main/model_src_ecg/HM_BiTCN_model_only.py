import torch
import torch.nn as nn
import torch.nn.functional as F
from .HM_BiTCN import *

class HM_BiTCN_Model(nn.Module):
    def __init__(self, input_channels, complexity, num_classes=16, dilations=[8, 4, 2, 1], mix_weight=0.3, dropout=0.1):
        super(HM_BiTCN_Model, self).__init__()
        self.mix_weight = mix_weight

        self.input_proj = nn.Conv2d(input_channels, complexity, kernel_size=1, bias=False)
        self.bn_input = nn.BatchNorm2d(complexity)

        self.hidden_mix = nn.Conv2d(in_channels=complexity,
                                    out_channels=complexity,
                                    kernel_size=1,
                                    bias=False)
        self.bn_hidden = nn.BatchNorm2d(complexity)

        self.encoder = HM_BiTCN_Encoder(complexity, dilations)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(complexity, num_classes)

    def forward(self, x):
        
        x = self.input_proj(x)
        x = self.bn_input(x)
        x = F.leaky_relu(x, 0.2)
        # HM block
        mix = self.hidden_mix(x)
        mix = self.bn_hidden(mix)
        x = x + self.mix_weight * mix
        x = x.squeeze(2)  
        # BiTCN encoder
        x = self.encoder(x)
        x = self.dropout(x)
        x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)
        x = self.projection(x)
        return x
    def predict(self, x):
        """Sigmoid-based multi-label prediction"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs
