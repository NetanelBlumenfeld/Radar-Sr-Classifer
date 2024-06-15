import os

import torch
from gestures.configs import new_config as cfg1
from gestures.data_loader2.dataset_factory import get_data_loader

# from gestures.data_loader.tiny_data_loader import get_tiny_data_loader
from gestures.network.experiment_tracker import get_time_in_string
from gestures.network.runner import Runner
from gestures.setup import setup_callbacks, setup_model

if __name__ == "__main__":

    data_config = cfg1.data_config
    main_config = cfg1.main_config
    data_dir, output_dir, device = (
        main_config["data_dir"],
        main_config["output_dir"],
        main_config["device"],
    )
    files = os.listdir("/Users/netanelblumenfeld/Downloads/11G/test")
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
        "RandomGesture",
    ]
    base_dir = "/Users/netanelblumenfeld/Downloads/11G/test"

    data_loader = get_data_loader(files, 5, gestures, base_dir)
    dummy_tensor = torch.randn(10, 10, device=device)

    # loop for different ds factors
    extra_info = "not_norm_doppler"

    # getting model
    model, optimizer, acc, loss_metric = setup_model(
        task=main_config["task"],
        model_cfg=cfg1.model_config,
        device=device,
    )

    # experiment name
    data_pre_name = (
        f"ds_{data_config['ds_factor']}_original_dim_{data_config['original_dims']}"
    )
    data_pre_name += f"_{extra_info}" if extra_info else ""
    experiment_name = os.path.join(
        main_config["task"],
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
        loader_train=data_loader,
        loader_validation=data_loader,
        loader_test=data_loader,
        device=device,
        optimizer=optimizer,
        loss_metric=loss_metric,
        acc_metric=acc,
        callbacks=callbacks,
        base_dir=base_dir,
        task=main_config["task"],
    )
    runner.run(epochs=cfg1.epochs)
