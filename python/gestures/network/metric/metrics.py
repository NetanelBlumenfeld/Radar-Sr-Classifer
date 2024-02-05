import torch
import torch.nn as nn
from pytorch_msssim import MS_SSIM, SSIM


class Psnr(nn.Module):
    def __init__(self):
        super(Psnr, self).__init__()

    def forward(self, trues, preds):
        mse = torch.mean((trues - preds) ** 2)
        if mse == 0:
            # MSE is zero means no noise is present in the signal.
            # Therefore, PSNR has no importance.
            return torch.tensor(100.0)
        max_pixel = 1
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr


class Msssim(torch.nn.Module):
    def __init__(self):
        super(Msssim, self).__init__()
        self.name = "msssim"
        self.ms = MS_SSIM(
            data_range=1.0,
            size_average=True,
            channel=2,
            win_size=2,
            nonnegative_ssim=True,
            K=(0.21, 0.45),
        )

    def forward(self, img1, img2):
        return self.ms(img1, img2)


class Ssim(torch.nn.Module):
    def __init__(self):
        super(Ssim, self).__init__()
        self.name = "sssim"
        self.ssim = SSIM(
            data_range=1.0,
            size_average=True,
            channel=2,
            win_size=2,
            nonnegative_ssim=True,
            K=(0.21, 0.45),
        )

    def forward(self, img1, img2):
        return self.ssim(img1, img2)
