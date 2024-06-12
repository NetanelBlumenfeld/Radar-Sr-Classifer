import os

import numpy as np
from gestures.data_loader1.data_loader_factory import (
    ClassifierDataset,
    SrClassifierDataset,
    SrDataset,
)
from torch.utils.data import DataLoader


def load_data_set(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load data from the data directory.

    Params:
    * data_dir: path to the data directory
    :return: a tuple of data and labels
    """
    data_path = os.path.join(data_dir, "X.npy")
    label_path = os.path.join(data_dir, "y.npy")
    X = np.load(data_path)
    y = np.load(label_path)
    return X, y


def get_data_loader(
    data_dir: str, task: str, loader_cfg: dict, processing_func: dict, ds_factor: int
) -> dict[str, DataLoader]:
    """
    Returns a dict that contain a data loaders name and the data loader for train/val/test.

    Params:
    * data_dir: path to the data directory
    * loaders: list of data loaders to use if it None it will load all the data loaders
    * processed_data: list of function to process the data before training
    * use_pool: whether to use multiprocessing pool to load the data
    :return: a tuple of train and test data loaders
    """
    res = {"train": None, "val": None, "test": None}
    for loader in loader_cfg["loaders"]:
        shuffle_enable = True if loader == "train" else False
        data_set_dir = os.path.join(data_dir, loader)
        X, y = load_data_set(data_set_dir)
        if ds_factor == 8:
            lenX = X.shape[-1]
            X = X[:, :, :, :, 2 : lenX - 2]
        if task == "classifier":
            dataset = ClassifierDataset(
                dataX=X,
                dataY=y,
                pre_train_process=processing_func["pre_train_process"],
                on_train_process=processing_func["on_train_process"],
            )
        elif task == "sr":
            dataset = SrDataset(
                dataX=X,
                pre_train_process=processing_func["pre_train_process"],
                hr_to_lr=processing_func["hr_to_lr"],
            )

        elif task == "sr_classifier":
            dataset = SrClassifierDataset(
                dataX=X,
                dataY=y,
                pre_train_process=processing_func["pre_train_process"],
                hr_to_lr=processing_func["hr_to_lr"],
                on_train_process=processing_func["on_train_process"],
            )
        else:
            raise ValueError(f"Unknown task: {task}")
        res[loader] = DataLoader(
            dataset, batch_size=loader_cfg["batch_size"], shuffle=shuffle_enable
        )
    return res
