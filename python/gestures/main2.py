import os

import torch
from gestures.configs import new_config as cfg1
from gestures.data_loader2.dataset_factory import get_data_loader, get_data_loader_multi
from gestures.network.callbacks.callback_logger import get_time_in_string
from gestures.network.runner_multi import Runner
from gestures.setup import (
    get_pc_cgf,
    setup_callbacks,
    setup_model,
    setup_model_multi,
    setup_model_rec,
)

if __name__ == "__main__":

    pc, data_dir, output_dir, device = get_pc_cgf()
    task = "sr_classifier"  # task = ["sr", "classifier", "sr_classifier"]
    original_dims = True if task == "classifier" else False
    batch_size = 3
    epochs = 400

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

    data_loader = get_data_loader_multi(task, batch_size, gestures, data_dir)
    dummy_tensor = torch.randn(10, 10, device=device)

    # getting model
    model, optimizer, acc, loss_metric = setup_model_rec(
        task=task,
        model_cfg=cfg1.model_config,
        device=device,
    )

    # experiment name
    data_pre_name = "multi_scales"
    experiment_name = os.path.join(
        task,
        f"{model.model_name}_{loss_metric.name}",
        data_pre_name,
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
