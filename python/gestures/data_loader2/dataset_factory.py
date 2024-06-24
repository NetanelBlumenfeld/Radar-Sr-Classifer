import os
from typing import Dict

import numpy as np
import torch
from gestures.utils_processing_data import (
    ComplexToRealOneSample,
    DopplerMapOneSample,
    DownSampleOneSample,
    NormalizeOneSample,
    ToTensor,
)
from torch.utils.data import DataLoader, Dataset


def construct_label(
    file_name: str, gestures: list[str], duplication_number: int
) -> torch.Tensor:

    label = file_name.split("_")[1]
    label = gestures.index(label)
    label = torch.tensor([label] * duplication_number).flatten()
    return label


class BasicDataset(Dataset):
    def __init__(self, files: list[str], base_dir: str):
        self.files = files
        self.base_dir = base_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = np.load(os.path.join(self.base_dir, file))
        data = ToTensor()(data)
        return data


class ClassifierDataset(Dataset):
    def __init__(
        self, files: list[str], gestures: list[str], base_dir: str, pre_processing_funcs
    ):
        self.files = files
        self.gestures = gestures
        self.base_dir = base_dir
        self.pre_processing_funcs = pre_processing_funcs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = self.pre_processing_funcs(np.load(os.path.join(self.base_dir, file)))
        label = construct_label(file, self.gestures, data.shape[0])
        return (data, label)

    # def process_data(self, data):
    #     data = ToTensor()(data)
    #     data = NormalizeOneSample()(data)
    #     data = DopplerMapOneSample()(data)
    #     return data


class SrClassifierDataset(Dataset):
    def __init__(self, files: list[str], gestures: list[str], base_dir: str):
        self.files = files
        self.gestures = gestures
        self.base_dir = base_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = np.load(os.path.join(self.base_dir, file))
        low_res_data, high_res_data = self.process_data(data)
        label = construct_label(file, self.gestures, data[0].shape[0])
        return low_res_data, (high_res_data, label)

    def process_data(
        self, high_res_data: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        low_res_data = ToTensor()(high_res_data)
        low_res_data = DownSampleOneSample(dx=16, dy=4, original_dims=True)(
            low_res_data
        )
        low_res_data = NormalizeOneSample()(low_res_data)
        low_res_data = ComplexToRealOneSample()(low_res_data)

        high_res_data_tensor = ToTensor()(high_res_data)
        high_res_data_tensor = NormalizeOneSample()(high_res_data_tensor)
        high_res_data_tensor = ComplexToRealOneSample()(high_res_data_tensor)
        return low_res_data, high_res_data_tensor


def get_data_loader(
    task: str, batch_size: int, gestures: list[str], base_dir: str, pre_processing_funcs
) -> dict[str, DataLoader]:
    data_loaders: Dict[str, DataLoader] = {}

    # for data_kind in ["tt"]:
    for data_kind in ["train", "val", "test"]:
        data_dir = os.path.join(base_dir, data_kind)
        files = os.listdir(data_dir)
        files = [file for file in files if file.endswith(".npy")]

        if task == "classifier":
            data_set = ClassifierDataset(
                files, gestures, data_dir, pre_processing_funcs
            )
        elif task == "sr_classifier":
            data_set = SrClassifierDataset(files, gestures, data_dir)
        else:
            raise ValueError("Unknown task: " + task)
        shuffle = True if data_kind == "train" else False

        data_loaders[data_kind] = DataLoader(
            data_set, batch_size=batch_size, shuffle=shuffle
        )

    return data_loaders


if __name__ == "__main__":
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
    k = 1
    base_dir = "/Users/netanelblumenfeld/Downloads/11G/test"
    data_loader = get_data_loader("classification", files, 5, gestures, base_dir)
