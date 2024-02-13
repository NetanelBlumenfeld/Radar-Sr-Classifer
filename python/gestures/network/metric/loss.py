import torch
import torch.nn as nn
from gestures.network.metric.metrics import Msssim, Ssim


class LossFunctionTinyRadarNN:
    def __init__(
        self,
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


class MsssimLoss(torch.nn.Module):
    def __init__(self, window_size=5, size_average=True, channel=2):
        super(MsssimLoss, self).__init__()
        self.msssim = Msssim()
        self.name = "msssim_loss"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, img1, img2):
        ms_ssim = self.msssim(img1, img2)
        return 1 - ms_ssim


class SsimLoss(torch.nn.Module):
    def __init__(self, window_size=5, size_average=True, channel=2):
        super(SsimLoss, self).__init__()
        self.ssim = Ssim()
        self.name = "ssim_loss"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, img1, img2):
        return 1 - self.ssim(img1, img2)
