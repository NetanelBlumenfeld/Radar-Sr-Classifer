from gestures.network.models.classifiers.tiny_radar import TinyRadarNN
from gestures.network.models.super_resolution.drln import Drln
from gestures.network.models.super_resolution.safmn import SAFMN

tiny = {
    "model_cls": TinyRadarNN,
    "model_cfg": {},
    # "model_ck": "/home/netanel/code/outputs1/classifier/TinyRadar_loss_TinyLoss/ds_1_original_dim_True_not_norm_doppler/2024-03-04_12:59:50/model/total_loss.pth",
    "model_ck": None,
}

drln = {
    "model_cls": Drln,
    "model_cfg": {"num_drln_blocks": 2, "scale": 4, "num_channels": 2},
    "model_ck": None,
}

safmn = {
    "model_cls": SAFMN,
    "model_cfg": {
        "dim": 36,
        "n_blocks": 8,
        "ffn_scale": 2.0,
        "upscaling_factor": 4,
        "channels": 2,
    },
    # "model_ck": "/home/netanel/code/outputs1/sr/SAFMN_loss_L1/ds_4_original_dim_False_not_norm_doppler/2024-03-07_17:57:12/model/total_loss.pth",
    "model_ck": None,
}
