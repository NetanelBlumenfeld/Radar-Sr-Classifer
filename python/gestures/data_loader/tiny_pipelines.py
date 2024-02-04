from typing import Callable

import numpy as np
from scipy.fftpack import fft, fftshift


def doppler_map(x: np.ndarray, ax: int = 1) -> np.ndarray:
    """input shape is (N,doppler_points,range_points)"""
    if ax == 1:
        assert x.ndim == 3, f"ax is {ax} so data must be 3D"
    if ax == 0:
        assert x.ndim == 2, f"ax is {ax} so data must be 2D"
    return np.abs(fftshift(fft(x, axis=ax), axes=ax))


def set_low_res(img: np.ndarray, down_sp_func, dims: tuple[int, int, int]) -> list[int]:
    low_res = down_sp_func(img)
    return [dims[0], dims[1], dims[2], low_res.shape[1], low_res.shape[2]]


# def sr_classifier_time_4090_pipeline(
#     X: tuple[np.ndarray, np.ndarray], ds_func: Callable, norm_func: Callable
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     data: (N,5,2,32,492)
#     """
#     data, labels = X
#     result_high_res = np.zeros(data.shape, dtype=np.complex64)
#     low_res_dim = set_low_res(data[0, 0], ds_func, data.shape[:3])
#     result_low_res = np.zeros(low_res_dim, dtype=np.complex64)
#     for sample in range(data.shape[0]):
#         for time_step in range(data.shape[1]):
#             for sensor in range(data.shape[2]):
#                 high_res = norm_func(data[sample, time_step, sensor])
#                 result_high_res[sample, time_step, sensor] = high_res
#                 result_low_res[sample, time_step, sensor] = ds_func(high_res)
#     result_low_res = np.stack((result_low_res.real, result_low_res.imag), axis=3)
#     result_high_res = np.stack((result_high_res.real, result_high_res.imag), axis=3)

#     return result_low_res, result_high_res, labels


def sr_classifier_time_4090_pipeline(
    X: tuple[np.ndarray, np.ndarray], ds_func: Callable, norm_func: Callable
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    data: (N,5,2,32,492)
    """
    data, labels = X
    high_res_real = norm_func(data.real)
    high_res_imag = norm_func(data.imag)
    high_res = np.stack((high_res_real, high_res_imag), axis=3)
    low_res = high_res[:, :, :, :, ::4, ::4]

    return low_res, high_res, labels


def sr_classifier_4090_pipeline(
    X: tuple[np.ndarray, np.ndarray], ds_func: Callable, norm_func: Callable
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data, labels = X
    result_high_res = np.zeros(data.shape, dtype=np.float32)
    low_res_dim = set_low_res(data[0, 0], ds_func, data.shape[:3])

    result_low_res = np.zeros(low_res_dim, dtype=np.float32)
    for sample in range(data.shape[0]):
        for time_step in range(data.shape[1]):
            high_res = doppler_map(data[sample, time_step])
            result_high_res[sample, time_step] = norm_func(high_res)

            low_res_time = ds_func(data[sample, time_step])
            result_low_res[sample, time_step] = norm_func(doppler_map(low_res_time))

    return result_low_res, result_high_res, labels


def sr_time_4090_pipeline(
    X: tuple[np.ndarray, np.ndarray], ds_func: Callable, norm_func: Callable
) -> tuple[np.ndarray, np.ndarray]:
    low_res, high_res, _ = sr_classifier_time_4090_pipeline(X, ds_func, norm_func)
    dims = high_res.shape
    assert dims[1:] == (5, 2, 2, 32, 492)
    high_res = high_res.reshape(dims[0] * dims[1] * dims[2], 2, dims[4], dims[5])
    dims = low_res.shape
    assert dims[1:4] == (5, 2, 2)
    low_res = low_res.reshape(dims[0] * dims[1] * dims[2], 2, dims[4], dims[5])

    return low_res, high_res


def classifier_pipeline(
    data: np.ndarray, norm_func, ds_func, time_domain: bool
) -> tuple[np.ndarray, np.ndarray]:
    X, labels = data
    res = np.zeros(X.shape, dtype=np.float32)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                data = X[i, j, k]
                data = norm_func(data)
                data = ds_func(data)

                if time_domain:
                    data = doppler_map(data, ax=0)
                data = norm_func(data)
                res[i, j, k] = data
    return res, labels
