from enum import Enum

import torch.nn as nn
from gestures.network.metric.loss import LossFunctionTinyRadarNN, MsssimLoss, SsimLoss
from gestures.network.metric.metrics import Msssim, Psnr, Ssim


class LossType(Enum):
    L1 = "L1"
    MSE = "MSE"
    CrossEntropy = "CrossEntropy"
    Huber = "Huber"
    MsssimLoss = "MsssimLoss"
    SsimLoss = "SsimLoss"
    SSIM = "SSIM"
    MSSSIM = "MSSSIM"
    PSNR = "PSNR"
    TinyLoss = "TinyLoss"


class LossFactory:
    loss_functions = {
        "L1": nn.L1Loss(),
        "MSE": nn.MSELoss(),
        "huber": nn.HuberLoss(delta=0.65),
        "CrossEntropy": nn.CrossEntropyLoss(),
        "MsssimLoss": MsssimLoss(),
        "SsimLoss": SsimLoss(),
        "SSIM": Ssim(),
        "MSSSIM": Msssim(),
        "PSNR": Psnr(),
        "TinyLoss": LossFunctionTinyRadarNN(),
    }

    @staticmethod
    def get_loss_function(name):
        if name not in LossFactory.loss_functions:
            raise ValueError(f"Loss function '{name}' not recognized")
        return LossFactory.loss_functions[name]
