import torch
import torch.nn as nn
from gestures.network.metric.ssim import MS_SSIM, SSIM


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
            K=(0.01, 0.45),
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
            K=(0.1, 0.45),
        )

    def forward(self, img1, img2):
        return self.ssim(img1, img2)


class ClassificationAccuracy(nn.Module):
    def __init__(self, numberOfGestures: int = 12):
        super(ClassificationAccuracy, self).__init__()
        self.numberOfGestures = numberOfGestures
        self.name = "classification_accuracy"

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        pred = outputs.reshape(-1, self.numberOfGestures).max(1)
        squashed_labels = labels.reshape(-1)
        total = squashed_labels.shape[0]
        correct = pred[1].eq(squashed_labels).sum().item()
        return correct, total


class AccuracyMetric:

    def __init__(self):
        self.metric_function = ClassificationAccuracy()
        self.name = "classification"
        self.values = []
        self.running_total = 0

    @property
    def value(self):
        # return {"acc": 100 * (sum(self.values) / self.running_total)}
        return sum(self.values) / (self.running_total)

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        correct, total = self.metric_function(outputs, labels)

        self.values.append(correct)
        self.running_total += total

    def reset(self):
        self.values = []
        self.running_total = 0.0
