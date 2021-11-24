# Author: Little-Chen
# Emial: Chenxiuyan_t@163.com

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class MY_CNN(nn.Module):
    def __init__(self):
        super(MY_CNN, self).__init__()

        self.model = nn.Sequential( #3 32 32
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2),# 6 32 32
            nn.MaxPool2d(kernel_size=2, stride=2, padding =0),#6 16 16
            nn.BatchNorm2d(num_features=6),
            nn.CELU(inplace=True),
            
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1, padding=2),# 12 16 16
            nn.MaxPool2d(kernel_size=2, stride=2, padding =0), #12 8 8
            nn.BatchNorm2d(num_features=12),
            nn.CELU(inplace=True),
            
            nn.Conv2d(in_channels=12, out_channels=8, kernel_size=3, stride=1, padding=1),# 8 8 8
            nn.BatchNorm2d(num_features=8),
            nn.CELU(inplace=True),

            nn.Conv2d(in_channels=8, out_channels=6, kernel_size=3, stride=1, padding=1),# 6 8 8
            nn.BatchNorm2d(num_features=6),
            nn.CELU(inplace=True),

            nn.Conv2d(in_channels=6, out_channels=4, kernel_size=3, stride=1, padding=1),# 4 8 8
            nn.MaxPool2d(kernel_size=2, stride=2, padding =0), # 4 4 4
            nn.BatchNorm2d(num_features=4),
            nn.CELU(inplace=True),
            )

        self.liner = nn.Sequential(

            nn.Linear(4*4*4,4*4*4),
            nn.CELU(inplace=True),

            nn.Linear(4*4*4,16),
            nn.Tanh(),

            nn.Linear(16,4)
        )
        
    def forward(self, img):
        
        x = self.model(img)

        z = self.liner(x.view(x.shape[0], -1))

        return z
