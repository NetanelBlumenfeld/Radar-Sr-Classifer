import gestures.configs.new_config_models as model_cfg
import torch
from gestures.network.metric.criterion_factory import MetricCriterion
from gestures.network.models.sr_classifier.SRCnnTinyRadar import (
    CombinedSRDrlnClassifier,
)
from gestures.setup import get_pc_cgf
from gestures.utils_processing_data import (
    ComplexToRealOneSample,
    DopplerMapOneSample,
    DownSampleOneSample,
    NormalizeOneSample,
    ToTensor,
)

# pc configs
pc, data_dir, output_dir, device = get_pc_cgf()

# data configs
task = "sr_classifier"  # task = ["sr", "classifier", "sr_classifier"]
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
data_config = {
    "ds_factor_x": 4,
    "ds_factor_y": 4,
    "original_dims": True if task == "classifier" else False,
}
pre_processing_funcs = {
    "classifier": torch.nn.Sequential(
        DownSampleOneSample(), ToTensor(), NormalizeOneSample(), DopplerMapOneSample()
    ),
    "sr_classifier": {
        "hr": torch.nn.Sequential(
            ToTensor(), NormalizeOneSample(), ComplexToRealOneSample()
        ),
        "lr": torch.nn.Sequential(
            ToTensor(),
            DownSampleOneSample(),
            NormalizeOneSample(),
            ComplexToRealOneSample(),
        ),
    },
}

epochs = 5

lr = 0.0015
