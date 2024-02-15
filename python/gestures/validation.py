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
    extra_info = "not_norm_doppler"
    # dp_cfg["ds_factor"] = ds

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
