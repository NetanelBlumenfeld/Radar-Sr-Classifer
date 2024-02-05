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


class MsssimLoss(torch.nn.Module):
    def __init__(self, window_size=5, size_average=True, channel=2):
        super(MsssimLoss, self).__init__()
        self.msssim = Msssim()
        self.name = "msssim_loss"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, img1, img2):
        return 1 - self.msssim(img1, img2)


class SsimLoss(torch.nn.Module):
    def __init__(self, window_size=5, size_average=True, channel=2):
        super(SsimLoss, self).__init__()
        self.ssim = Ssim()
        self.name = "ssim_loss"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, img1, img2):
        return 1 - self.ssim(img1, img2)
