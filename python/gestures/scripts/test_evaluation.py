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


class A(torch.nn.Module):
    def __init__(self):
        super(A, self).__init__()

    @staticmethod
    def reshape_to_model_output(low_res, high_res, device):
        high_res_imgs = high_res.permute(1, 0, 2, 3, 4, 5).to(device)
        sequence_length, batch_size, sensors, channels, H, W = high_res_imgs.size()
        new_batch = sequence_length * batch_size * sensors
        high_res_imgs = high_res_imgs.reshape(new_batch, channels, H, W)

        low_res = low_res.permute(1, 0, 2, 3, 4, 5).to(device)
        sequence_length, batch_size, sensors, channels, H, W = low_res.size()
        new_batch = sequence_length * batch_size * sensors
        low_res = low_res.reshape(new_batch, channels, H, W)
        return low_res.to(device), high_res_imgs.to(device)

    def forward(self, x):
        return x


if __name__ == "__main__":
    model_path = "/Users/netanelblumenfeld/Desktop/bgu/Msc/code/outputs1/classifier/TinyRadar_loss_TinyLoss/ds_4_original_dim_True_not_norm_doppler/2024-03-04_12:41:39/model/total_loss.pth"

    pc, data_dir, output_dir, device = get_pc_cgf()
    task = "classifier"  # task = ["sr", "classifier", "sr_classifier"]
    original_dims = True if task == "classifier" else False
    for x in [1]:
        for dim in [36]:
            batch_size = 30
            dx, dy = 4, 4
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
                        DownSampleOneSample(dx=dx, dy=dy, original_dims=True),
                        NormalizeOneSample(),
                        ComplexToRealOneSample(),
                    ),
                },
                "sr": {
                    "hr": torch.nn.Sequential(
                        ToTensor(), NormalizeOneSample(), ComplexToRealOneSample()
                    ),
                    "lr": torch.nn.Sequential(
                        ToTensor(),
                        DownSampleOneSample(dx=dx, dy=dy, original_dims=True),
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
            # model, _, _, _ = BasicModel.load_pre_train_model(device, model_path)
            model = A()

            loss_metric.reset()
            acc.reset()
            validate(model, data_loader["test"], device, loss_metric, acc)
            print(acc.value)
            print(loss_metric.value)
