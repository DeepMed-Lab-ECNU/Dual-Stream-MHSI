import torch
import torch.nn as nn

class RTGB(nn.Module):
    def __init__(self, out_channel, height, width):
        super(RTGB, self).__init__()
        self.channel_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
        self.wide_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(height, height, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.high_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width, width, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        c = self.channel_gap(x)
        h = self.high_gap(x.transpose(-3, -2))
        w = self.wide_gap(x.transpose(-3, -1))
        # h = h.transpose(-3, -2)
        # w = w.transpose(-3, -1)
        ch = torch.einsum("...cbd,...hbd->...chbd", c, h)
        chw = torch.einsum("...chbd,...wbd->...chw", ch, w)
        return chw

class DRTLM(nn.Module):
    def __init__(self, rank, out_channel, height, width):
        super(DRTLM, self).__init__()
        self.rank = rank
        self.preconv = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=1),
                                     nn.BatchNorm2d(out_channel),
                                     nn.ReLU(inplace=True))
        self.rtgb = RTGB(out_channel, height, width)
        self.projection = nn.Conv2d(out_channel*rank, out_channel, 1)

    def resblock(self, input):
        xup = self.rtgb(input)
        res = input - xup
        return xup, res

    def forward(self, input):

        x = self.preconv(input)
        (xup, xdn) = self.resblock(x)
        temp_xup = xdn
        output = xup
        for i in range(1, self.rank):
            (temp_xup, temp_xdn) = self.resblock(temp_xup)
            output = torch.cat((output, temp_xup), 1)
            temp_xup = temp_xdn

        output = self.projection(output) * x
        output = input + output

        return output