from enum import Enum
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM


class LossType(Enum):
    L1 = "L1"
    MSE = "MSE"
    CrossEntropy = "CrossEntropy"
    Huber = "Huber"
    MSSSIM = "MSSSIM"
    TinyLoss = "TinyLoss"


class SimpleLoss:
    def __init__(self, loss_function: LossType):
        self.loss_function = LossFactory.get_loss_function(loss_function.name.lower())
        self.name = loss_function.name

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        compute loss

        Args:
            outputs (torch.Tensor): the outputs from TinyRadarNN model
            labels (torch.Tensor): labels for the data

        Returns:
            loss (float): the loss
        """
        return self.loss_function(outputs, labels)


class LossFunctionTinyRadarNN:
    def __init__(
        self,
        loss_function: LossType = LossType.CrossEntropy,
        numberOfTimeSteps: int = 5,
    ):
        self.numberOfTimeSteps = numberOfTimeSteps
        self.loss_function = nn.CrossEntropyLoss()
        self.name = "tiny_cross_entropy"

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        compute loss for tiny radar classifier

        Args:
            outputs (torch.Tensor): the outputs from TinyRadarNN model
            labels (torch.Tensor): labels for the data

        Returns:
            loss (float): the loss
        """
        loss = 0
        for i in range(self.numberOfTimeSteps):
            loss += self.loss_function(outputs[i], labels[i])
        return loss / self.numberOfTimeSteps


class LossFunctionSRTinyRadarNN:
    def __init__(
        self,
        loss_type_srcnn,
        loss_type_classifier,
        wight_srcnn: float = 0.5,
        wight_classifier: float = 0.5,
    ):
        self.loss_func_srcnn = loss_type_srcnn
        self.loss_func_classifier = loss_type_classifier
        self.wight_srcnn = wight_srcnn
        self.wight_classifier = wight_classifier
        self.name = f"sr_{loss_type_srcnn.name}_classifier_{loss_type_classifier.name}"

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        compute loss for tiny radar classifier

        Args:
            outputs (torch.Tensor): the outputs from TinyRadarNN model
            labels (torch.Tensor): labels for the data

        Returns:
            loss (float): the loss
        """
        # high_res_pred = outputs[0].reshape(-1, 2, 32, 492)
        # high_res_true = labels[0].reshape(-1, 2, 32, 492)
        # loss_srcnn = self.loss_func_srcnn.update(high_res_pred, high_res_true)
        loss_classifier = self.loss_func_classifier.update(outputs[1], labels[1])
        # loss = self.wight_srcnn * loss_srcnn + self.wight_classifier * loss_classifier
        return loss_classifier, loss_classifier, 0


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=5, size_average=True, channel=2):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.name = "msssim"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window = self.create_window(window_size, self.channel, self.device)

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.33**2

        l_p = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        cs_p = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        ssim_map = l_p * cs_p

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def gaussian_window(self, size, sigma):
        gauss = torch.Tensor(
            [exp(-((x - size // 2) ** 2) / float(2 * sigma**2)) for x in range(size)]
        )
        return gauss / gauss.sum()

    def create_window(self, window_size, channel, device):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window.to(device)

    def forward(self, img1, img2):
        weights = [0.2, 0.45, 0.35]
        levels = len(weights)
        msssim = 1.0
        for level in range(levels):
            sim = self.ssim(
                img1,
                img2,
                self.window,
                self.window_size,
                self.channel,
                self.size_average,
            )
            msssim *= sim ** weights[level]
            print(f"level {level} sim = {sim}, msssim = {msssim}")

            if level < levels - 1:
                img1 = F.avg_pool2d(img1, (2, 2))
                img2 = F.avg_pool2d(img2, (2, 2))

        return 1 - msssim


class MSS(torch.nn.Module):
    def __init__(self):
        super(MSS, self).__init__()
        self.name = "ms"
        self.ms = MS_SSIM(
            data_range=1.0,
            size_average=True,
            channel=2,
            win_size=2,
            nonnegative_ssim=True,
            K=(0.11, 0.5),
        )

    def forward(self, img1, img2):
        loss = 1 - self.ms(img1, img2)
        return loss


class LossFactory:
    loss_functions = {
        "L1": nn.L1Loss(),
        "MSE": nn.MSELoss(),
        "huber": nn.HuberLoss(delta=0.65),
        "CrossEntropy": nn.CrossEntropyLoss(),
        "MSSSIM": MSS(),
        "TinyLoss": LossFunctionTinyRadarNN(),
    }

    @staticmethod
    def get_loss_function(name):
        if name not in LossFactory.loss_functions:
            raise ValueError(f"Loss function '{name}' not recognized")
        return LossFactory.loss_functions[name]
