import math
from torch import nn
import torch
from math import sqrt


class ESPCN(nn.Module):
    def __init__(self, scale_factor=4, num_channels=1):
        super(ESPCN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, padding=5//2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=3 // 2),
            nn.PReLU()
        )
        self.last_part = nn.Sequential(
            nn.Conv2d(32, num_channels * (scale_factor** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(scale_factor)
        )

        self._initialize_weights()

    def _initialize_weights(m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             if m.in_channels == 32:
    #                 nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
    #                 nn.init.zeros_(m.bias.data)
    #             else:
    #                 nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
    #                 nn.init.zeros_(m.bias.data)
    # def _initialize_weights(self):
    #     """
    #     Initial the weights.
    #     """
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):  # 判断一个量是否是相应的类型
    #             # m.weight.data.normal_(0.0, 0.2)
    #             m.weight.data.normal_(0.0, sqrt(2 / m.out_channels / m.kernel_size[0] / m.kernel_size[0]))  # MSRA
    #             # nn.init.xavier_normal(m.weight) # Xavier
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #
    # def forward(self, x):
    #     out = self.first_part(x)
    #     out = self.mid_part(out)
    #     out = self.last_part(out)
    #     return out
    def forward(self, x):
        # x1 = self.first_part(x)
        # x2 = self.last_part(x1)
        # x3 = self.last_part(x2)
        x1 = self.first_part(x)

        x3 = self.last_part(x1)
        return x3



