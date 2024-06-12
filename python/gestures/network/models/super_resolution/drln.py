import torch
import torch.nn as nn
import torch.nn.init as init
from gestures.network.models import custom_layers as ops
from gestures.network.models.basic_model import BasicModel


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = ops.BasicBlock(channel, channel // reduction, 3, 1, 3, 3)
        self.c2 = ops.BasicBlock(channel, channel // reduction, 3, 1, 5, 5)
        self.c3 = ops.BasicBlock(channel, channel // reduction, 3, 1, 7, 7)
        self.c4 = ops.BasicBlockSig((channel // reduction) * 3, channel, 3, 1, 1)

    def forward(self, x):
        y = self.avg_pool(x)
        c1 = self.c1(y)
        c2 = self.c2(y)
        c3 = self.c3(y)
        c_out = torch.cat([c1, c2, c3], dim=1)
        y = self.c4(c_out)
        return x * y


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()

        self.r1 = ops.ResidualBlock(in_channels, out_channels)
        self.r2 = ops.ResidualBlock(in_channels * 2, out_channels * 2)
        self.r3 = ops.ResidualBlock(in_channels * 4, out_channels * 4)
        self.g = ops.BasicBlock(in_channels * 8, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        c0 = x

        r1 = self.r1(c0)
        c1 = torch.cat([c0, r1], dim=1)

        r2 = self.r2(c1)
        c2 = torch.cat([c1, r2], dim=1)

        r3 = self.r3(c2)
        c3 = torch.cat([c2, r3], dim=1)

        g = self.g(c3)
        out = self.ca(g)
        return out


class DrlnBlock(nn.Module):
    def __init__(self, chs):
        super(DrlnBlock, self).__init__()
        self.b1 = Block(chs, chs)
        self.b2 = Block(chs, chs)
        self.b3 = Block(chs, chs)
        self.c1 = ops.BasicBlock(chs * 2, chs, 3, 1, 1)
        self.c2 = ops.BasicBlock(chs * 3, chs, 3, 1, 1)
        self.c3 = ops.BasicBlock(chs * 4, chs, 3, 1, 1)

    def forward(self, x):
        b1 = self.b1(x)
        c1 = torch.cat([x, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        return o3 + x


class Drln(BasicModel):
    def __init__(self, num_drln_blocks: int = 2, scale: int = 4, num_channels: int = 2):
        self.model_name = f"Drln_{num_drln_blocks}"
        super(Drln, self).__init__(self.model_name)
        self.scale = scale
        chs = 64
        self.head = nn.Conv2d(num_channels, chs, 3, 1, 1)
        # Kaiming Initialization
        init.kaiming_normal_(self.head.weight, mode="fan_out", nonlinearity="relu")
        if self.head.bias is not None:
            init.constant_(self.head.bias, 0)
        self.upsample = ops.UpsampleBlock(chs, self.scale, multi_scale=False)
        # self.convert = ops.ConvertBlock(chs, chs, 20)
        self.tail = nn.Conv2d(chs, num_channels, 3, 1, 1)
        drln_blocks = [DrlnBlock(chs) for _ in range(num_drln_blocks)]
        self.drln_blocks = nn.Sequential(*drln_blocks)

    @staticmethod
    def reshape_to_model_output(low_res, high_res, device):
        d0, d1, d2, d3, d4 = low_res.shape
        low_res = low_res.reshape(d0 * d1, d2, d3, d4)
        d0, d1, d2, d3, d4 = high_res.shape
        high_res = high_res.reshape(d0 * d1, d2, d3, d4)
        return low_res.to(device), high_res.to(device)

    def forward(self, x):
        x = self.head(x)
        c0 = o0 = x
        o0 = self.drln_blocks(o0)
        b_out = o0 + c0
        out = self.upsample(b_out, scale=self.scale)
        out = self.tail(out)
        return out
