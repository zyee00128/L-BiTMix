import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class LinearAttention(nn.Module):
    """
    Linear Attention
    (Q * K^T) * V -> Q * (K^T * V)
    """
    def __init__(self, feature_dim):
        super(LinearAttention, self).__init__()
        self.feature_dim = feature_dim
        self.q_proj = nn.Conv1d(feature_dim, feature_dim, 1)
        self.k_proj = nn.Conv1d(feature_dim, feature_dim, 1)
        self.v_proj = nn.Conv1d(feature_dim, feature_dim, 1)
        self.eps = 1e-4
        self.scale = feature_dim ** -0.5 
        self.bn = nn.BatchNorm1d(feature_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0)
    def forward(self, x):
        
        # x shape: [Batch, Channel, Length] -> [Batch, Length, Channel]
        input_x = self.bn(x)
        
        Q = self.q_proj(input_x) # [B, C, L]
        K = self.k_proj(input_x) # [B, C, L]
        V = self.v_proj(input_x) # [B, C, L]
        Q = Q * self.scale
        K = K * self.scale
        Q = F.elu(Q) + 1.0
        K = F.elu(K) + 1.0

        # (B, L, C) -> (B, L, 1)
        KV = torch.bmm(V, K.permute(0, 2, 1)) 
        Q_perm = Q.permute(0, 2, 1) # [B, L, C]
        numerator = torch.bmm(Q_perm, KV.permute(0, 2, 1))
        # Z = 1 / (Q * \sum K^T)
        K_sum = K.sum(dim=-1, keepdim=True) # (B, C, 1)
        denom = torch.bmm(Q_perm, K_sum)
        denom = torch.clamp(denom, min=self.eps)
        output = numerator / denom
        output = output.permute(0, 2, 1)
        output = output + x

        return output


class TemporalBlock(nn.Module):
    """
    Dilated Causal Conv -> WeightNorm -> ReLU -> Dropout
    """
    def __init__(self, input_channels, complexity, dilation, dropout=0.1):
        super(TemporalBlock, self).__init__()
        kernel_size = 9
        self.padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(input_channels, complexity, kernel_size, 
                            stride=1, padding=0, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(complexity)
        self.conv2 = nn.Conv1d(complexity, complexity, kernel_size, 
                            stride=1, padding=0, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(complexity)
        self.attention = LinearAttention(complexity)
        self.downsample = None
        if input_channels != complexity:
            self.downsample = nn.Conv1d(input_channels, complexity, 1)
            self.bn3 = nn.BatchNorm1d(complexity)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_pad1 = F.pad(x, (self.padding, 0))
        y = self.conv1(x_pad1)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = self.attention(y)
        # out = y

        x_pad2 = F.pad(out, (self.padding, 0))
        out = self.conv2(x_pad2)
        out = self.bn2(out)
        out = self.relu(out)    
        out = self.dropout(out)

        if self.downsample is not None:
            res = self.downsample(x)
            res = self.bn3(res)
        else:
            res = x
        return self.relu(out + res)

class ECG_TCN(nn.Module):
    """
    Input -> Conv -> TCN Blocks -> Linear Attention -> Linear Classifier
    """
    def __init__(self, input_channels, complexity, num_classes, num_layers=8, dropout=0.1):
        super(ECG_TCN, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(input_channels, complexity, 1),
            nn.BatchNorm1d(complexity),
            nn.ReLU())
        layers = []
        for i in range(num_layers):
            dilation_size = 2 ** i
            # Dilated Convolution
            layers += [TemporalBlock(complexity, complexity,
                                     dilation=dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.attention = LinearAttention(complexity)
        self.bn_final = nn.BatchNorm1d(complexity)

        # 全局池化
        self.gap = nn.AdaptiveAvgPool1d(1)
        # 分类头
        self.linear = nn.Linear(complexity, num_classes)

    def forward(self, x):
        x = x.squeeze(2)
        y = self.embedding(x)
        # TCN 
        y = self.tcn(y)
        # y = self.bn_final(y)
        # y = self.attention(y)
        y = self.gap(y).squeeze(-1)
        y = self.linear(y)
        return y