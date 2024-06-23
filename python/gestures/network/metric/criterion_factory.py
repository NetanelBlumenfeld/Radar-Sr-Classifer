from enum import Enum

import torch.nn as nn
from gestures.network.metric.custom_criterion import (
    AccuracyMetric,
    LossFunctionTinyRadarNN,
    Msssim,
    MsssimLoss,
    Psnr,
    Ssim,
    SsimLoss,
)


class MetricCriterion(Enum):
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
    ClassifierAccuracy = "ClassifierAccuracy"


class CriterionFactory:

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
        "ClassifierAccuracy": AccuracyMetric(),
    }

    @staticmethod
    def get_loss_function(name):
        if name not in CriterionFactory.loss_functions:
            raise ValueError(f"Loss function '{name}' not recognized")
        return CriterionFactory.loss_functions[name]
