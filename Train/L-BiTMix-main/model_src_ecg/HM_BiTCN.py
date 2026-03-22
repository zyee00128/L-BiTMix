import torch
import torch.nn as nn
import torch.nn.functional as F

class HM_BiTCN_Block(nn.Module):
    """Hidden-Mix Bidirectional Temporal Convolutional Network"""
    def __init__(self, channels, kernel_size=3, dilation_forward=1, dilation_backward=1, groups=1):
        super().__init__()
        self.padding_forward = (kernel_size - 1) * dilation_forward
        self.padding_backward = (kernel_size - 1) * dilation_backward
        self.conv_forward = nn.Conv1d(channels, channels, kernel_size, 
                                      padding=0, dilation=dilation_forward, groups=groups)
        self.conv_backward = nn.Conv1d(channels, channels, kernel_size, 
                                       padding=0, dilation=dilation_backward, groups=groups)
        self.hidden_mix = nn.Conv1d(channels, channels, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        
        x_forward = F.pad(x, (self.padding_forward, 0))
        y_forward = self.conv_forward(x_forward)
        y_forward = y_forward[:, :, :x.size(2)]
       
        x_backward = torch.flip(x, [-1])
        x_backward = F.pad(x_backward, (self.padding_backward, 0))
        y_backward = self.conv_backward(x_backward)
        y_backward = torch.flip(y_backward, [-1])
        y_backward = y_backward[:, :, :x.size(2)]

        #combine
        y = y_forward + y_backward
        # y = self.act(y) ##for abluation study wo CM
        y = self.act(self.hidden_mix(y))
        return x + y

class HM_BiTCN_Encoder(nn.Module):
    """HM-BiTCN encoder"""
    def __init__(self, channels, dilations=[8, 4, 2, 1]):
        super().__init__()
        self.layers = nn.ModuleList([HM_BiTCN_Block(channels, dilation_forward=d, dilation_backward=d) 
                                    for d in dilations])

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x

## unidirectional TCNs
class TCN_Block(nn.Module):
    """Temporal Convolutional Network Block"""
    def __init__(self, channels, kernel_size=3, dilation=1, groups=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(channels, channels, kernel_size, 
                              padding=0, dilation=dilation, groups=groups)
        self.hidden_mix = nn.Conv1d(channels, channels, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        
        x_padded = F.pad(x, (self.padding, 0))
        y = self.conv(x_padded)
        y = y[:, :, :x.size(2)]
        y = self.act(self.hidden_mix(y))
        return x + y
class TCN_Encoder(nn.Module):
    def __init__(self, channels, dilations=[8, 4, 2, 1]):
        super().__init__()
        self.layers = nn.ModuleList([TCN_Block(channels, dilation=d) 
                                    for d in dilations])
    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x