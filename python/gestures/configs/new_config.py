import gestures.configs.new_config_models as model_cfg
import torch
from gestures.network.metric.metric_factory import LossType
from gestures.network.models.sr_classifier.SRCnnTinyRadar import (
    CombinedSRDrlnClassifier,
)
from gestures.setup import get_pc_cgf
from gestures.utils_preprocessing import (
    ComplexToReal,
    DopplerMaps,
    DownSampleOffLine,
    Normalize,
    ToTensor,
)

gestures = [
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
]

pc, data_dir, output_dir, device = get_pc_cgf()
task = "sr_classifier"  # task = ["sr", "classifier", "sr_classifier"]
epochs = 200
lr = 0.0015


main_config = {
    "pc": pc,
    "task": task,
    "data_dir": data_dir,
    "output_dir": output_dir,
    "device": device,
}

data_config = {
    "ds_factor": 2,
    "original_dims": True if task == "classifier" else False,
}

loaders = ["test"] if pc == "mac" else ["test"]
data_loader = {
    "data_dir": data_dir,
    "batch_size": 20,
    "loaders": loaders,
}
pr_funcs = {
    "pre_train_process": torch.nn.Sequential(
        Normalize(),
        ToTensor(),
        ComplexToReal(),
        DownSampleOffLine(D=1, original_dim=data_config["original_dims"]),
        # DopplerMaps(),
    ),
    "on_train_process": None,
    "hr_to_lr": torch.nn.Sequential(
        Normalize(),
        ToTensor(),
        ComplexToReal(),
        DownSampleOffLine(D=data_config["ds_factor"]),
    ),
}


#### Section For Models ####
_classifier = {
    "model": model_cfg.tiny["model_cls"],
    "model_cfg": model_cfg.tiny["model_cfg"],
    "model_ck": model_cfg.tiny["model_ck"],
    "loss1": [{"metric": LossType.TinyLoss, "wight": 1}],
    "optimizer": {"class": torch.optim.Adam, "args": {"lr": lr}},
    "accuracy_metric": [LossType.ClassifierAccuracy],
}
_sr = {
    "model": model_cfg.safmn["model_cls"],
    "model_cfg": model_cfg.safmn["model_cfg"],
    "model_ck": model_cfg.safmn["model_ck"],
    "loss": [
        {"metric": LossType.L1, "wight": 0.5},
        {"metric": LossType.MSSSIM, "wight": 0.5},
    ],
    "accuracy_metric": [LossType.PSNR, LossType.MSE, LossType.MSSSIM],
    "optimizer": {"class": torch.optim.Adam, "args": {"lr": lr}},
}
_sr_classifier_models = {
    "model": {
        "sr": _sr,
        "classifier": _classifier,
        "combined": CombinedSRDrlnClassifier,
    },
    "loss": {
        "sr": {"wight": 0.5, "loss_type": LossType.L1},
        "classifier": {"wight": 1, "loss_type": LossType.TinyLoss},
    },
    "optimizer": {"class": torch.optim.Adam, "args": {"lr": lr}},
}

model_config = {
    "sr_classifier": _sr_classifier_models,
    "classifier": _classifier,
    "sr": _sr,
}


#### Section For Callbacks ####
_tensor_board = {
    "need_dir": True,
    "active": True,
    "args": {
        "base_dir": None,
        "classes_name": gestures,
        "with_cm": False if task == "sr" else True,
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
