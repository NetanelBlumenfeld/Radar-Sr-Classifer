import os

import torch
from gestures.configs import new_config as cfg1
from gestures.data_loader2.dataset_factory import get_data_loader
from gestures.network.callbacks.callback_logger import get_time_in_string
from gestures.network.models.basic_model import BasicModel
from gestures.network.runner2 import Runner, validate
from gestures.setup import get_pc_cgf, setup_callbacks, setup_model
from gestures.utils_processing_data import (
    ComplexToRealOneSample,
    DopplerMapOneSample,
    DownSampleOneSample,
    NormalizeOneSample,
    ToTensor,
)

if __name__ == "__main__":
    model_path = "/Users/netanelblumenfeld/Desktop/bgu/Msc/code/out_laptop/sr_classifier/sr_SAFMN_frozen_classifier_TinyRadar_sr_loss_L10.5_classifier_loss_TinyLoss1/dsx_4_dsy_4_original_dim_False/2024-07-11_09:37:13/model/total_loss.pth"

    pc, data_dir, output_dir, device = get_pc_cgf()
    task = "sr_classifier"  # task = ["sr", "classifier", "sr_classifier"]
    original_dims = True if task == "classifier" else False
    for x in [1]:
        for dim in [24, 36]:
            batch_size = 30
            dx, dy = 1, 4
            epochs = 70

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
            pre_processing_funcs = {
                "classifier": torch.nn.Sequential(
                    ToTensor(),
                    DownSampleOneSample(dx=dx, dy=dy, original_dims=original_dims),
                    NormalizeOneSample(),
                    DopplerMapOneSample(),
                ),
                "sr_classifier": {
                    "hr": torch.nn.Sequential(
                        ToTensor(), NormalizeOneSample(), ComplexToRealOneSample()
                    ),
                    "lr": torch.nn.Sequential(
                        ToTensor(),
                        DownSampleOneSample(dx=dx, dy=dy, original_dims=original_dims),
                        NormalizeOneSample(),
                        ComplexToRealOneSample(),
                    ),
                },
            }

            data_loader = get_data_loader(
                task, batch_size, gestures, data_dir, pre_processing_funcs[task]
            )

            # getting model
            qq, optimizer, acc, loss_metric = setup_model(
                task=task,
                model_cfg=cfg1.model_config,
                device=device,
            )
            model, _, _, _ = BasicModel.load_pre_train_model(device, model_path)

            loss_metric.reset()
            acc.reset()
            validate(model, data_loader["test"], device, loss_metric, acc)
            print(acc.value)
            print(loss_metric.value)
