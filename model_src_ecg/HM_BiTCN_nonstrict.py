import torch
import torch.nn as nn
import torch.nn.functional as F

class HM_BiTCN_Block(nn.Module):
    """Hidden-Mix Bidirectional Temporal Convolutional Network"""
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv_forward = nn.Conv1d(channels, channels, kernel_size, padding=self.padding, dilation=dilation)
        self.conv_backward = nn.Conv1d(channels, channels, kernel_size, padding=self.padding, dilation=dilation)
        self.hidden_mix = nn.Conv1d(channels, channels, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        ## 1
        y_forward = self.conv_forward(x)[:, :, :x.size(2)]
        y_backward_full = self.conv_backward(torch.flip(x, [-1]))
        y_backward = torch.flip(y_backward_full, [-1])[:, :, :x.size(2)]
        ## 2
        # x_fwd = F.pad(x, (self.padding, 0))
        # y_forward = self.conv_forward(x_fwd)
        # y_forward = y_forward[:, :, :x.size(2)]
        
        # x_bwd = torch.flip(x, [-1])
        # x_bwd = F.pad(x_bwd, (self.padding, 0))
        # y_backward = self.conv_backward(x_bwd)
        # y_backward = torch.flip(y_backward, [-1])
        # y_backward = y_backward[:, :, :x.size(2)]

        #combine
        y = y_forward + y_backward
        y = self.act(self.hidden_mix(y))
        return x + y

class HM_BiTCN_Encoder(nn.Module):
    """HM-BiTCN encoder"""
    def __init__(self, channels, layers=[8, 4, 2, 1]):
        super().__init__()
        self.layers = nn.ModuleList([HM_BiTCN_Block(channels, dilation=d) for d in layers])

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x
