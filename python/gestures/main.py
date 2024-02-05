import os

from gestures import config as cfg
from gestures.data_loader.tiny_data_loader import get_tiny_data_loader
from gestures.network.experiment_tracker import get_time_in_string
from gestures.network.runner import Runner
from gestures.setup import get_pc_cgf, setup_callbacks, setup_model

if __name__ == "__main__":
    pc = cfg.pc
    data_dir, output_dir, device = get_pc_cgf(pc)

    # loop for different ds factors
    dp_cfg = cfg.data_preprocessing_cfg
    for ds in [1, 2, 4, 8]:
        # extra_info = f"ds_{ds}"
        dp_cfg["ds_factor"] = ds

        # getting data loaders
        data_cfg = cfg.data_cfg
        trainloader, valloader, testloader = get_tiny_data_loader(
            data_dir=data_dir,
            data_scg=data_cfg,
            data_preprocessing_cfg=dp_cfg,
            use_pool=False,
            batch_size=cfg.batch_size,
            # threshold=-1,
        )
        # getting model
        model, optimizer, acc, loss_metric = setup_model(
            task=dp_cfg["task"],
            model_cfg=cfg.model_cfg,
            device=device,
        )

        # experiment name
        data_pre_name = f"ds_{dp_cfg['ds_factor']}_original_dim_{dp_cfg['original_dims']}_pix_norm_{dp_cfg['pix_norm']}"
        data_pre_name += f"_th_{extra_info}" if extra_info else ""
        experiment_name = os.path.join(
            dp_cfg["task"],
            f"{model.model_name}_{loss_metric.name}",
            data_pre_name,
            get_time_in_string(),
        )

        # callbacks
        base_dir = os.path.join(output_dir, experiment_name)
        callbacks = setup_callbacks(cfg.callbacks_cfg, base_dir=base_dir)

        # #training
        training_cfg = cfg.training_cfg
        runner = Runner(
            model=model,
            loader_train=trainloader,
            loader_validation=valloader,
            loader_test=testloader,
            device=device,
            optimizer=optimizer,
            loss_metric=loss_metric,
            acc_metric=acc,
            callbacks=callbacks,
            base_dir=base_dir,
            task=dp_cfg["task"],
        )
        runner.run(epochs=training_cfg["epochs"])
