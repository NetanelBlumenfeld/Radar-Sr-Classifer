import torch
from gestures.data_loader.tiny_data_loader import Normalization
from gestures.network.metric.metric_factory import LossType
from gestures.network.models.classifiers.tiny_radar import TinyRadarNN
from gestures.network.models.sr_classifier.SRCnnTinyRadar import (
    CombinedSRDrlnClassifier,
)
from gestures.network.models.super_resolution.drln import Drln
from gestures.network.models.super_resolution.safmn import SAFMN

pc = "4090"
if pc == "4090":
    batch_size = 32
    people = 26
elif pc == "mac":
    people = 2
    batch_size = 4
elif pc == "3080":
    people = 26
    batch_size = 32
else:
    raise ValueError("Unknown pc")

_ds_scale_factor = 4
data_type = "npy"
lr = 0.0015
# data config
data_cfg = {
    "gestures": [
        "PinchIndex",
        "PinchPinky",
        "FingerSlider",
        "FingerRub",
        "SlowSwipeRL",
        "FastSwipeRL",
        "Push",
        "Pull",
        "PalmTilt",
        "Circle",
        "PalmHold",
        "NoHand",
    ],
    "people": people,
    "data_type": data_type,
}
# training config
training_cfg = {"lr": lr, "epochs": 150, "batch_size": batch_size}


# data preprocessing config
data_preprocessing_cfg = {
    "task": "sr_classifier",  # "sr", "classifier", "sr_classifier"
    "time_domain": True if data_type == "npy" else False,
    "processed_data": True,
    "pix_norm": Normalization.Range_0_1,
    "ds_factor": _ds_scale_factor,
    "original_dims": False,
}

_classifier = {
    "model": TinyRadarNN,
    "model_cfg": {},
    # "model_ck": "/home/netanel/code/outputs/classifier/TinyRadar_loss_TinyLoss_1/ds_1_original_dim_True_pix_norm_Normalization.Range_0_1_th_not_norm_doppler/2024-02-10_16:54:26/model/acc.pth",  # noqa
    "model_ck": None,
    "loss1": [{"metric": LossType.TinyLoss, "wight": 1}],
    "optimizer": {"class": torch.optim.Adam, "args": {"lr": lr}},
    "accuracy": [LossType.ClassifierAccuracy],
}
_sr = {
    "model": Drln,
    "model_cfg": {"num_drln_blocks": 2, "scale": _ds_scale_factor, "num_channels": 2},
    # "model_ck": "/home/netanel/code/outputs/sr/Drln_2_loss_L1_1/ds_4_original_dim_False_pix_norm_Normalization.Range_0_1/2024-02-09_15:03:43/model/loss.pth",  # noqa
    "model_ck": "/Users/netanelblumenfeld/Desktop/bgu/Msc/code/outputs/sr/Drln_2_loss_L1_1/ds_4_original_dim_False_pix_norm_Normalization.Range_0_1/2024-02-09_15:03:43/model/loss.pth",  # noqa
    "loss": {
        "metrics_names": [LossType.L1, LossType.MsssimLoss],
        "metric_wights": [0.5, 0.5],
    },
    "loss1": [
        {"metric": LossType.L1, "wight": 1},
    ],
    "acc": [LossType.L1, LossType.MSE],
    "optimizer": {"class": torch.optim.Adam, "args": {"lr": lr}},
    "accuracy": {
        "metrics_names": [LossType.PSNR, LossType.SSIM],
        "metric_wights": [1, 1],
    },
}
_sr_classifier_models = {
    "model": {
        "sr": _sr,
        "classifier": _classifier,
        "combined": CombinedSRDrlnClassifier,
    },
    "loss": {
        "sr": {"wight": 0, "loss_type": LossType.L1},
        "classifier": {"wight": 1, "loss_type": LossType.TinyLoss},
    },
    "acc": [LossType.ClassifierAccuracy, LossType.MSE, LossType.PSNR, LossType.SSIM],
    "optimizer": {"class": torch.optim.Adam, "args": {"lr": lr}},
}

model_cfg = {
    "sr_classifier": _sr_classifier_models,
    "classifier": _classifier,
    "sr": _sr,
}


# call backs config
_tensor_board = {
    "need_dir": True,
    "active": True,
    "args": {
        "base_dir": None,
        "classes_name": data_cfg["gestures"],
        "with_cm": False if data_preprocessing_cfg["task"] == "sr" else True,
    },
}
_save_model = {
    "need_dir": True,
    "active": True,
    "args": {
        "base_dir": None,
        "save_best": True,
        "metrics": ["val_total_loss"],
        "opts": ["min"],
    },
}
mode = 0
_progress = {
    "active": False,
    "args": {"verbose": mode},
    "need_dir": True if mode == 2 else False,
}  # 0 - progress bar, 1 - print, 2 - logger
_lr_scheduler = {
    "active": False,
    "need_dir": False,
    "args": {"gamma": 0.995, "optimizer": torch.optim.Adam},
}
_logger = {
    "active": True,
    "need_dir": True,
    "args": {"base_dir": None},
}
callbacks_cfg = {
    "tensor_board": _tensor_board,
    "save_model": _save_model,
    "progress": _progress,
    "lr_scheduler": _lr_scheduler,
    "logger": _logger,
}
