import os

import torch
from gestures.configs import new_config as cfg1
from gestures.data_loader2.dataset_factory import get_data_loader
from gestures.network.callbacks.callback_logger import get_time_in_string
from gestures.network.runner2 import Runner
from gestures.setup import get_pc_cgf, setup_callbacks, setup_model
from gestures.utils_processing_data import (
    ComplexGaussianNoiseTransform,
    ComplexToRealOneSample,
    DopplerMapOneSample,
    DownSampleOneSample,
    NormalizeOneSample,
    PipeLine,
    ToTensor,
)

if __name__ == "__main__":

    pc, data_dir, output_dir, device = get_pc_cgf()
    task = "sr_classifier"  # task = ["sr", "classifier", "sr_classifier"]
    original_dims = True if task == "classifier" else False
    for x in [1]:
        for dim in [36]:
            batch_size = 30
            dx, dy = 4, 4
            epochs = 250

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
            class_pipe = PipeLine(
                [
                    ToTensor(),
                    DownSampleOneSample(dx=dx, dy=dy, original_dims=original_dims),
                    NormalizeOneSample(),
                    DopplerMapOneSample(),
                ]
            )
            sr_pipe_hr = PipeLine(
                [ToTensor(), NormalizeOneSample(), ComplexToRealOneSample()]
            )
            sr_pipe_lr = PipeLine(
                [
                    ToTensor(),
                    ComplexGaussianNoiseTransform(),
                    DownSampleOneSample(dx=dx, dy=dy, original_dims=original_dims),
                    NormalizeOneSample(),
                    ComplexToRealOneSample(),
                ]
            )
            pre_processing_funcs = {
                "classifier": class_pipe,
                "sr_classifier": {
                    "hr": sr_pipe_hr,
                    "lr": sr_pipe_lr,
                },
            }

            data_loader = get_data_loader(
                task, batch_size, gestures, data_dir, pre_processing_funcs[task]
            )
            dummy_tensor = torch.randn(10, 10, device=device)

            # getting model
            model, optimizer, acc, loss_metric = setup_model(
                task=task,
                model_cfg=cfg1.model_config,
                device=device,
            )
            # loss_metric.sr_weight = gamma
            model.drln.dim = dim

            experiment_name = os.path.join(
                task,
                f"{model.model_name}",
                f"loss_{loss_metric.name}",
                f"{sr_pipe_lr.name}",
                get_time_in_string(),
            )
            print(experiment_name)

            # callbacks
            base_dir = os.path.join(output_dir, experiment_name)
            callbacks = setup_callbacks(cfg1.callbacks_cfg, base_dir=base_dir)

            # #training
            runner = Runner(
                model=model,
                loader_train=data_loader["train"],
                loader_validation=data_loader["val"],
                loader_test=data_loader["test"],
                device=device,
                optimizer=optimizer,
                loss_metric=loss_metric,
                acc_metric=acc,
                callbacks=callbacks,
                base_dir=base_dir,
                task=task,
            )
            runner.run(epochs=epochs)
