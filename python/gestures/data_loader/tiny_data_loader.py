from enum import Enum, auto
from functools import partial
from typing import Callable

import cv2
import numpy as np
from gestures.data_loader.tiny_data import load_tiny_data
from gestures.data_loader.tiny_data_factory import (
    classifier_loader,
    sr_classifier_loader,
    sr_loader,
)
from torch.utils.data import DataLoader


def down_sample_img(
    x: np.ndarray, row_factor: int, col_factor: int, original_dim: bool = False
) -> np.ndarray:
    def _down_sample(img: np.ndarray, row_factor: int, col_factor: int) -> np.ndarray:
        return img[::row_factor, ::col_factor]

    def _up_scale(img: np.ndarray, dim_up: tuple[int, int]) -> np.ndarray:
        if img.dtype == np.complex64:
            real_img = np.real(img)
            imag_img = np.imag(img)
            data_real_up = cv2.resize(real_img, dim_up, interpolation=cv2.INTER_CUBIC)
            data_imag_up = cv2.resize(imag_img, dim_up, interpolation=cv2.INTER_CUBIC)
            return data_real_up + 1j * data_imag_up
        else:
            return cv2.resize(img, dim_up, interpolation=cv2.INTER_CUBIC)

    assert x.ndim == 2
    org_dim = (x.shape[1], x.shape[0])
    img = _down_sample(x, row_factor, col_factor)
    if original_dim:
        img = _up_scale(img, org_dim)

    return img


def down_sample_data_sr(
    x: np.ndarray, row_factor: int, col_factor: int, original_dim: bool = False
) -> np.ndarray:
    if x.ndim == 2:
        return down_sample_img(x, row_factor, col_factor, original_dim)
    assert x.ndim == 3
    if original_dim:
        res = np.empty_like(x)
    else:
        res = np.empty(
            (
                x.shape[0],
                x.shape[1] // row_factor,
                x.shape[2] // col_factor,
            ),
            dtype=np.complex64,
        )
    x_len = x.shape[0]
    for i in range(x_len):
        res[i] = down_sample_img(x[i], row_factor, col_factor, original_dim)
    return res


class Normalization(Enum):
    NONE = auto()
    Range_0_1 = auto()
    Range_neg_1_1 = auto()


def normalize_img(img: np.ndarray, pix_norm: Normalization) -> np.ndarray:
    EPSILON = 1e-8

    if pix_norm == Normalization.NONE:
        return img
    elif pix_norm == Normalization.Range_0_1:
        return (img - np.min(img)) / (np.max(img) - np.min(img) + EPSILON)
    elif pix_norm == Normalization.Range_neg_1_1:
        return (img - np.min(img)) / (np.max(img) - np.min(img) + EPSILON) * 2 - 1
    else:
        raise ValueError("Unknown normalization type: " + str(pix_norm))


def get_pipeline_function(
    task: str, pix_norm: Normalization, ds_factor: int = 4, original_dims: bool = False
) -> tuple[Callable, Callable]:
    def _identity(x: np.ndarray) -> np.ndarray:
        return x

    norm_func = partial(normalize_img, pix_norm=pix_norm)
    if ds_factor != 1:
        if task == "classifier":
            ds_func = partial(
                down_sample_img,
                row_factor=ds_factor,
                col_factor=ds_factor,
                original_dim=original_dims,
            )
        else:
            ds_func = partial(
                down_sample_data_sr,
                row_factor=ds_factor,
                col_factor=ds_factor,
                original_dim=original_dims,
            )
    else:
        ds_func = _identity
    return norm_func, ds_func


def get_tiny_data_loader(
    data_dir: str,
    data_scg: dict,
    data_preprocessing_cfg: dict,
    use_pool: bool = True,
    batch_size: int = 32,
    # threshold: float = 0.0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns a data loader for the tiny dataset.

    Params:
    * data_dir: path to the data directory
    * processed_data: whether the data should be processed before setting the data loader
    * use_pool: whether to use multiprocessing pool to load the data
    :return: a tuple of train and test data loaders
    """
    dataset_loader_dict = {
        "sr": sr_loader,
        "classifier": classifier_loader,
        "sr_classifier": sr_classifier_loader,
    }
    task = data_preprocessing_cfg["task"]
    if task not in dataset_loader_dict:
        raise ValueError(f"Unknown task: {task}")
    dataset_func = dataset_loader_dict[task]
    dataset_func_partial = partial(dataset_func, use_pool=use_pool)
    X, y = load_tiny_data(
        data_dir=data_dir,
        people=data_scg["people"],
        gestures=data_scg["gestures"],
        data_type=data_scg["data_type"],
        use_pool=use_pool,
    )
    # X[X < threshold] = 0

    norm_func, ds_func = get_pipeline_function(
        task,
        data_preprocessing_cfg["pix_norm"],
        data_preprocessing_cfg["ds_factor"],
        data_preprocessing_cfg["original_dims"],
    )
    train_data_set, val_data_set, test_data_set = dataset_func_partial(
        X=X,
        labels=y,
        norm_func=norm_func,
        ds_func=ds_func,
        processed_data=data_preprocessing_cfg["processed_data"],
        time_domain=data_preprocessing_cfg["time_domain"],
    )
    trainloader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_data_set, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)
    return trainloader, valloader, testloader
