import os

import torch
from gestures.configs import new_config as cfg1
from gestures.data_loader2.dataset_factory import get_data_loader
from gestures.network.callbacks.callback_logger import get_time_in_string
from gestures.network.runner2 import Runner
from gestures.setup import get_pc_cgf, setup_callbacks, setup_model
from gestures.utils_processing_data import (
    ComplexToRealOneSample,
    DopplerMapOneSample,
    DownSampleOneSample,
    NormalizeOneSample,
    ToTensor,
)

if __name__ == "__main__":

    pc, data_dir, output_dir, device = get_pc_cgf()
    task = "sr_classifier"  # task = ["sr", "classifier", "sr_classifier"]
    original_dims = True if task == "classifier" else False
    for x, y in [
        # (1, 1),
        # (1, 2),
        # (2, 1),
        # (2, 2),
        # (4, 4),
        (4, 8),
        # (8, 4),
        # (2, 4),
        # (4, 2),
        # (8, 2),
        # (2, 8),
    ]:
        batch_size = 20
        dx, dy = x, y
        epochs = 100

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
        dummy_tensor = torch.randn(10, 10, device=device)

        # getting model
        model, optimizer, acc, loss_metric = setup_model(
            task=task,
            model_cfg=cfg1.model_config,
            device=device,
        )

        # experiment name
        data_pre_name = f"dsx_{dx}_dsy_{dy}_original_dim_{original_dims}"
        experiment_name = os.path.join(
            task,
            f"{model.model_name}_{loss_metric.name}",
            data_pre_name,
            get_time_in_string(),
        )

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
