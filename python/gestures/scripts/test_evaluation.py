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
    model_path = "/Users/netanelblumenfeld/Desktop/bgu/Msc/code/out/sr_classifier/rec/total_loss.pth"
    pc, data_dir, output_dir, device = get_pc_cgf()
    task = "sr_classifier"  # task = ["sr", "classifier", "sr_classifier"]
    original_dims = True if task == "classifier" else False
    for x in [1]:
        for dim in [36]:
            batch_size = 30
            dx, dy = 8, 8
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
                "sr": {
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
            # model = A()

            loss_metric.reset()
            acc.reset()
            validate(model, data_loader["test"], device, loss_metric, acc)
            print(acc.value)
            print(f"results for dx {dx} dy {dy} ")
            print(loss_metric.value)


"""
results for dx 2 dy 2 
{'sr_acc_PSNR': 16.209843890566223, 'sr_acc_MSE': 0.024199546085759915, 'sr_acc_MSSSIM': 0.8070774717316548, 'classifier_acc_ClassifierAccuracy': 0.8561897966159048}
{'total_loss': tensor(0.4360), 'sr_total_loss': 0.08700814418143944, 'sr_loss_L1': 0.08700814418143944, 'classifier_total_loss': 0.3924642714955406, 'classifier_loss_TinyLoss': 0.3924642714955406}
results for dx 8 dy 8
{'sr_acc_PSNR': 11.716476459372537, 'sr_acc_MSE': 0.06767164327790974, 'sr_acc_MSSSIM': 0.539056995481133, 'classifier_acc_ClassifierAccuracy': 0.47370674275869756}
{'total_loss': tensor(2.2908), 'sr_total_loss': 0.16144119491279643, 'sr_loss_L1': 0.16144119491279643, 'classifier_total_loss': 2.210086909500958, 'classifier_loss_TinyLoss': 2.210086909500958}

"""
