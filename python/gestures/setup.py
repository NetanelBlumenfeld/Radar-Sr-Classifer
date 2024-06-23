import os
import platform
from typing import Any

import torch as torch
from gestures.network.callbacks.callback_handler import CallbackHandler
from gestures.network.callbacks.callback_logger import Logger, get_time_in_string
from gestures.network.callbacks.callback_save_model import SaveModel
from gestures.network.callbacks.callback_tensotboard import BaseTensorBoardTracker
from gestures.network.metric.metric_tracker import (
    AccMetricTracker,
    AccMetricTrackerSrClassifier,
    LossMetricTracker,
    LossMetricTrackerSrClassifier,
)
from gestures.network.models.basic_model import BasicModel


def get_pc_cgf() -> tuple[str, str, str, torch.device]:
    # TODO: add option for 3080 and 4090
    os_info = platform.system()
    if os_info == "Linux":
        data_dir = "/mnt/netanelnew/tinyradar_data/"
        output_dir = "/home/netanel/code/out/"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pc = "4090"
    elif os_info == "Darwin":
        data_dir = "/Users/netanelblumenfeld/Downloads/11G"
        output_dir = "/Users/netanelblumenfeld/Desktop/bgu/Msc/code/outputs/"
        device = torch.device("cpu")
        pc = "mac"
    else:
        raise ValueError("OS must be Linux or Darwin")

    return pc, data_dir, output_dir, device


def classifier_model(model_cfg: dict, device: torch.device):
    """
    setup model,optimizer ,loss and acc metrics for training sr classifier

    Args:
        model_cfg (dict): _description_
    """
    # loss
    loss = LossMetricTracker(model_cfg["loss1"])

    # acc
    acc = AccMetricTracker(model_cfg["accuracy_metric"])

    # models
    model_cls = model_cfg["model"]
    if model_cfg["model_ck"]:
        model, _, _, _ = model_cls.load_model(device, model_cfg["model_ck"])
    else:
        model = model_cls(**model_cfg["model_cfg"]).to(device)

    # optimizer
    optimizer_class = model_cfg["optimizer"]["class"]
    optimizer_args = model_cfg["optimizer"]["args"]
    optimizer = optimizer_class(model.parameters(), **optimizer_args)
    return model, optimizer, acc, loss


def sr_model(model_cfg: dict, device: torch.device):
    """
    setup model,optimizer ,loss and acc metrics for training sr classifier

    Args:
        model_cfg (dict): _description_
    """
    loss = LossMetricTracker(model_cfg["loss"])
    acc = AccMetricTracker(model_cfg["accuracy_metric"])

    # models
    model_cls = model_cfg["model"]
    if model_cfg["model_ck"]:
        model, _, _, _ = model_cls.load_model(device, model_cfg["model_ck"])
    else:
        model = model_cls(**model_cfg["model_cfg"]).to(device)

    # optimizer
    optimizer_class = model_cfg["optimizer"]["class"]
    optimizer_args = model_cfg["optimizer"]["args"]
    optimizer = optimizer_class(model.parameters(), **optimizer_args)
    return model, optimizer, acc, loss


def sr_classifier_model(model_cfg: dict, device: torch.device):
    """
    setup model, loss and acc metrics for training sr classifier

    Args:
        model_cfg (dict): _description_
    """
    # models
    sr, _, acc_sr, loss_sr = sr_model(model_cfg["model"]["sr"], device)
    classifier, _, acc__classifier, loss_classifier = classifier_model(
        model_cfg["model"]["classifier"], device
    )
    model = model_cfg["model"]["combined"](sr, classifier).to(device)

    # loss
    loss_metric = LossMetricTrackerSrClassifier(
        sr_tracker=loss_sr,
        classifier_tracker=loss_classifier,
        sr_weight=model_cfg["loss"]["sr"]["wight"],
        classifier_weight=model_cfg["loss"]["classifier"]["wight"],
    )

    # acc
    acc_metric = AccMetricTrackerSrClassifier(
        sr_acc=acc_sr, classifier_acc=acc__classifier
    )
    # optimizer
    optimizer_class = model_cfg["optimizer"]["class"]
    optimizer_args = model_cfg["optimizer"]["args"]
    optimizer = optimizer_class(model.parameters(), **optimizer_args)
    return model, optimizer, acc_metric, loss_metric


def setup_model(
    task: str, model_cfg: dict, device: torch.device
) -> tuple[BasicModel, torch.optim.Optimizer, Any, LossMetricTrackerSrClassifier]:
    if task == "sr_classifier":
        model, optimizer, acc, loss_metric = sr_classifier_model(
            model_cfg[task], device
        )
    elif task == "sr":
        model, optimizer, acc, loss_metric = sr_model(model_cfg[task], device)

    elif task == "classifier":
        model, optimizer, acc, loss_metric = classifier_model(model_cfg[task], device)

    else:
        raise ValueError(f"Unknown task: {task}")
    return model, optimizer, acc, loss_metric


def setup_train_name(
    data_preprocessing_cfg: dict,
    loss: LossMetricTrackerSrClassifier,
    w_c: float,
    w_sr: float,
    train_cfg: dict,
    task: str,
    model_name: str,
) -> str:
    train_config = f"lr_{train_cfg['lr']}_batch_size_{train_cfg['batch_size']}_{loss.name}_w_sr_{w_sr}_w_c_{w_c}"  # noqa
    data_preprocess = f"ds_{data_preprocessing_cfg['ds_factor']}_original_dim_{data_preprocessing_cfg['original_dims']}_pix_norm_{data_preprocessing_cfg['pix_norm']}"  # noqa
    return os.path.join(
        task, model_name, data_preprocess, train_config, get_time_in_string()
    )


def setup_callbacks(callbacks_cfg: dict, base_dir: str) -> CallbackHandler:
    callbacks_dict = {
        "tensor_board": BaseTensorBoardTracker,
        # "progress": ProgressBar,
        "save_model": SaveModel,
        "lr_scheduler": torch.optim.lr_scheduler.ExponentialLR,
        "logger": Logger,
    }
    callbacks_list = []
    for callback_name, callback_args in callbacks_cfg.items():
        if callback_args["active"]:
            if callback_args["need_dir"]:
                callback_args["args"]["base_dir"] = base_dir
            callback = callbacks_dict[callback_name](**callback_args["args"])
            callbacks_list.append(callback)
    return CallbackHandler(callbacks_list)
