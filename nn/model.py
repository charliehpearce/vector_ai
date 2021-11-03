from os import stat
from numpy.lib.type_check import imag
from torch import nn
import numpy as np


class CovNet(nn.Module):
    def __init__(self, image_size) -> None:
        super(CovNet, self).__init__()

        s1 = self.conv_output_shape(np.array(image_size), kernel_size=5)
        s1pool = self.conv_output_shape(s1, kernel_size=2, stride=2)
        s2 = self.conv_output_shape(s1pool, kernel_size=5)
        s2pool = self.conv_output_shape(s2, kernel_size=2, stride=2)
        # output size * output channels
        self.output_dim = int(np.prod(s2pool) * 12)

        self.cov_layer1 = nn.Sequential(
            # First conv layer
            nn.Conv2d(1, 6, 5),  # n channels, output channels, kernel
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.cov_layer2 = nn.Sequential(
            # Second conv layer
            nn.Conv2d(6, 12, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.output_dim, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 60),
            nn.ReLU(inplace=True),
            nn.Linear(60, 10),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def conv_output_shape(h_w: np.array, kernel_size=1, stride=1, pad=0, dilation=1):
        return (((h_w-kernel_size) + 2*pad) / stride)+1

    def forward(self, x):
        x = self.cov_layer1(x)
        x = self.cov_layer2(x)
        x = x.view(-1, self.output_dim)  # flatten
        x = self.fc(x)
        return x
