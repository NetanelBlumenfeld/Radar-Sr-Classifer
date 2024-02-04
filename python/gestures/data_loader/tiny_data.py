import os
from functools import partial
from multiprocessing import Pool
from typing import Optional

import numpy as np


# script to load tiny data
def npy_feat_reshape(x: np.ndarray) -> np.ndarray:
    numberOfWindows = x.shape[0]
    numberOfSweeps = x.shape[1]
    numberOfRangePoints = x.shape[2]
    numberOfSensors = x.shape[3]
    lengthOfSubWindow = 32

    numberOfSubWindows = int(numberOfSweeps / lengthOfSubWindow)

    x = x.reshape(
        (
            numberOfWindows,
            numberOfSubWindows,
            lengthOfSubWindow,
            numberOfRangePoints,
            numberOfSensors,
        )
    )
    return x


def data_paths(
    data_dir: str, people: int, gestures: list[str], data_type: str
) -> list[str]:
    folder_path = os.path.join(data_dir, f"data_{data_type}")
    if data_type not in ["npy", "feat"]:
        raise ValueError("data type must be npy or doppler")

    file_suffix = "_1s.npy" if data_type == "npy" else "_1s_wl32_doppl.npy"
    return [
        os.path.join(folder_path, f"p{person}/{gesture}{file_suffix}")
        for person in range(1, people)
        for gesture in gestures
    ]


def set_labels(y: np.ndarray) -> np.ndarray:
    temp = y
    labels = np.empty((temp.shape[0], temp.shape[1]))
    for idx in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            for i in range(temp.shape[2]):
                if temp[idx][j][i] == 1:
                    labels[idx][j] = i
    del temp
    return labels


def load_data(
    data_path: str,
    data_type: str,
    gestures: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        data = np.load(data_path)
        if data_type == "npy":
            data = npy_feat_reshape(data)
    except Exception as e:
        raise ValueError(f"failed to load data from {data_path}: {str(e)}")
    SubjectLabel = []
    if gestures:
        gestureIdx = gestures.index(data_path.split("/")[-1].split("_")[0])
        gestures_num = len(gestures)
        for idx in range(0, data.shape[0]):
            GestureLabel = []
            for jdx in range(0, data.shape[1]):
                GestureLabel.append(np.identity(gestures_num)[gestureIdx])
            SubjectLabel.append(np.asarray(GestureLabel))
    return data, np.array(SubjectLabel)


def load_tiny_data(
    data_dir: str,
    people: int,
    gestures: list[str],
    data_type: str,
    use_pool: bool = True,
):
    paths = data_paths(data_dir, people, gestures, data_type)
    num_workers = os.cpu_count()
    load_data_func = partial(load_data, gestures=gestures, data_type=data_type)
    print(f"loading data with {num_workers} cpu cores")
    if use_pool:
        with Pool(num_workers) as p:
            data = p.map(load_data_func, paths)
    else:
        data = list(map(load_data_func, paths))
    y = set_labels(np.concatenate(list(map(lambda x: x[1], data))))
    X = np.concatenate(list(map(lambda x: x[0], data)))
    X = X.transpose(0, 1, 4, 2, 3)
    return X, y
