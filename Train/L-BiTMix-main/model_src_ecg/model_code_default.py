
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, BatchNorm1d, ReLU, AdaptiveAvgPool2d, Sequential, Dropout

from .Lora_layer_default import Conv2d, Linear
from .HM_BiTCN import *

if 'HM_BiTCN_info' not in globals():
    def HM_BiTCN_info(num_layers, complexity):
        return [None] * (num_layers + 20)

def Cutmix_student(x, y, device, alpha=0.75, valid_lead_num=12):
    return x, y
def Cutmix(x, y, device, alpha=0.75):
    return x, y, y, 1.0

class MyResidualBlock(nn.Module):
    def __init__(self, input_complexity, output_complexity, stride, downsample=False, rank_list=[1,1,1], **kwargs):
        super(MyResidualBlock, self).__init__()
        self.conv1 = Conv2d(input_complexity, output_complexity, (3, 3), stride=stride, padding=(1, 1), bias=False, r=rank_list[0])
        self.bn1 = BatchNorm2d(output_complexity)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv2d(output_complexity, output_complexity, (3, 3), stride=(1, 1), padding=(1, 1), bias=False, r=rank_list[1])
        self.bn2 = BatchNorm2d(output_complexity)
        
        self.downsample = Sequential(
            Conv2d(input_complexity, output_complexity, (1, 1), stride=stride, bias=False, r=rank_list[2]),
            BatchNorm2d(output_complexity)
        ) if downsample else None
        self.dropout = Dropout(0.1)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.dropout(self.relu(out + identity))

class NN_default_parallel(nn.Module):
    def __init__(self, nOUT, complexity=16, inputchannel=12, num_layers=14, rank_list=32, **kwargs):
        super(NN_default_parallel, self).__init__()
        if isinstance(rank_list, (int, float)):
            self.rank_list = [int(rank_list)] * (3 * (num_layers + 5))
        else:
            self.rank_list = rank_list

        self.first_layer = Sequential(
            nn.Conv1d(inputchannel, complexity, kernel_size=7, stride=2, padding=3, bias=False),
            BatchNorm1d(complexity),
            ReLU(True)
        )
        
        self.resnet_layers = nn.ModuleList()
        for i in range(num_layers + 1):
            stride = (1, 2) if i > 0 else (1, 1)
            self.resnet_layers.append(
                MyResidualBlock(complexity, complexity, stride, downsample=(stride != (1, 1)), rank_list=self.rank_list[3*i:3*i+3])
            )

        self.pool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(complexity, nOUT)

    def forward(self, x):
        if x.dim() == 4:
            if x.shape[2] == 1:
                x = x.squeeze(2)
            elif x.shape[3] == 1:
                x = x.squeeze(3)
                
        x = self.first_layer(x).unsqueeze(2)
        for layer in self.resnet_layers:
            x = layer(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

NN_default = NN_default_parallel
NN_default_series = NN_default_parallel
NN_default_replace = NN_default_parallel

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        return 1 - (2. * (inputs * targets).sum() + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)