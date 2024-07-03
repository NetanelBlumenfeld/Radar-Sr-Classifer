import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self.create_gaussian_kernel()

    def create_gaussian_kernel(self):
        k = self.kernel_size // 2
        x = torch.arange(-k, k + 1).float()
        y = torch.arange(-k, k + 1).float()
        xx, yy = torch.meshgrid(x, y)
        kernel = torch.exp(-0.5 * (xx**2 + yy**2) / self.sigma**2)
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        return kernel

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        kernel = self.kernel.to(x.device)
        kernel = kernel.expand(channels, 1, self.kernel_size, self.kernel_size)
        return F.conv2d(x, kernel, padding=self.kernel_size // 2, groups=channels)


class DownsampleConv2d(nn.Module):
    def __init__(
        self, in_channels=2, out_channels=2, kernel_size=3, stride=4, padding=1
    ):
        super(DownsampleConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            (2, 1),
            stride=(2, 1),
            padding=(0, 0),
        )

    def forward(self, x):
        return self.conv(x)


class DownsampleNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleNet, self).__init__()
        self.gaussian_blur = GaussianBlur(kernel_size=5, sigma=1.0)
        self.downsample_conv = DownsampleConv2d(
            in_channels, out_channels, kernel_size=3, stride=4, padding=1
        )

    def forward(self, x):
        x = self.gaussian_blur(x)
        x = self.downsample_conv(x)
        return x
