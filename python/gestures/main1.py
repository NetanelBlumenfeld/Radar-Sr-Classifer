import os

import torch
from gestures.configs import new_config as cfg1
from gestures.data_loader1.load import get_data_loader
from gestures.network.callbacks.callback_logger import get_time_in_string
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
    dummy_tensor = torch.randn(10, 10, device=device)

    # loop for different ds factors
    extra_info = "not_norm_doppler"

    data_loaders = get_data_loader(
        data_dir=main_config["data_dir"],
        task=main_config["task"],
        loader_cfg=cfg1.data_loader,
        processing_func=cfg1.pr_funcs,
        ds_factor=data_config["ds_factor"],
    )
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
        loader_train=data_loaders["test"],
        loader_validation=data_loaders["test"],
        loader_test=data_loaders["test"],
        device=device,
        optimizer=optimizer,
        loss_metric=loss_metric,
        acc_metric=acc,
        callbacks=callbacks,
        base_dir=base_dir,
        task=main_config["task"],
    )
    runner.run(epochs=cfg1.epochs)
