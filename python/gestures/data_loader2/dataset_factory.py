import os

import numpy as np
import torch
from gestures.utils_processing_data import (
    ComplexToRealOneSample,
    DopplerMapOneSample,
    DownSampleOneSample,
    NormalizeOneSample,
    SrMultiScaleProcessor,
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


class ClassifierDataset(Dataset):
    def __init__(self, files: list[str], gestures: list[str], base_dir: str):
        self.files = files
        self.gestures = gestures
        self.base_dir = base_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = self.process_data(np.load(os.path.join(self.base_dir, file)))
        label = construct_label(file, self.gestures, data.shape[0])
        return (data, label)

    def process_data(self, data):
        data = ToTensor()(data)
        data = NormalizeOneSample()(data)
        data = DopplerMapOneSample()(data)
        return data


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
        return low_res_data, [high_res_data, label]

    def process_data(self, high_res_data: np.ndarray) -> tuple[torch.Tensor]:
        low_res_data = ToTensor()(high_res_data)
        low_res_data = DownSampleOneSample(D=2)(low_res_data)
        low_res_data = NormalizeOneSample()(low_res_data)
        low_res_data = ComplexToRealOneSample()(low_res_data)

        high_res_data = ToTensor()(high_res_data)
        high_res_data = NormalizeOneSample()(high_res_data)
        high_res_data = ComplexToRealOneSample()(high_res_data)
        return low_res_data, high_res_data


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


def get_data_loader(files, batch_size, gestures, base_dir):
    data_set = SrClassifierDataset(files, gestures, base_dir)
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


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
    processor = SrMultiScaleProcessor(D=k)
    base_dir = "/Users/netanelblumenfeld/Downloads/11G/test"
    data_loader = get_data_loader(files, 5, gestures, base_dir)
    for i, (data, label) in enumerate(data_loader):
        process_data = processor(data, d=k)
        k *= 2
