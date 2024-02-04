from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class SrDataSet_3080(Dataset):
    def __init__(self, imgs: np.ndarray, transform: Callable) -> None:
        self.imgs = imgs
        self.transform = transform
        assert self.transform is not None

    def __len__(self) -> int:
        return self.imgs.shape[0]

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # high_res_time = self.imgs[idx]
        # high_res = self.transform(high_res_time)
        # low_res_time = down_sample_img(high_res_time, 4, 4)
        # low_res = self.transform(low_res_time)
        # return low_res, high_res
        high_res_time = self.transform(self.imgs[idx])
        low_res_time = np.zeros_like(high_res_time)
        low_res_time[0] = down_sample_img(high_res_time[0], 4, 4)
        low_res_time[1] = down_sample_img(high_res_time[1], 4, 4)
        return low_res_time, high_res_time


class ClassifierDataset(Dataset):
    def __init__(self, dataX, dataY):
        # self.x_train = np.transpose(dataX, (0, 1, 4, 2, 3))  # (N, 5, 2 ,32,492)
        self.x_train = dataX
        self.label = dataY

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x_train[idx], torch.LongTensor(self.label[idx])


class SrClassifierDataset_3080(Dataset):
    def __init__(
        self,
        hight_res: np.ndarray,
        labels: np.ndarray,
        transform: Callable,
    ) -> None:
        self.hight_res = np.transpose(hight_res, (0, 1, 4, 2, 3))  # (N, 5, 2 ,32,492)
        self.transform = transform
        self.tempy = labels
        self.label = np.empty((self.tempy.shape[0], self.tempy.shape[1]))
        print(self.label.shape)
        for idx in range(self.tempy.shape[0]):
            for j in range(self.tempy.shape[1]):
                for i in range(self.tempy.shape[2]):
                    if self.tempy[idx][j][i] == 1:
                        self.label[idx][j] = i
        del self.tempy

    def __len__(self) -> int:
        return self.hight_res.shape[0]

    def __getitem__(self, idx) -> tuple[np.ndarray, list]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        high_res_time = self.hight_res[idx]
        low_res, high_res = self.transform(high_res_time)
        return low_res, [high_res, torch.LongTensor(self.label[idx])]


class SrDataSet_4090(Dataset):
    def __init__(self, low_res, hight_res):
        """ "
        take input with shape (N,H,W) and add channel dim (N,1,H,W)
        """
        if low_res.ndim == 3:
            low_res = np.expand_dims(low_res, axis=1)
        if hight_res.ndim == 3:
            hight_res = np.expand_dims(hight_res, axis=1)
        self.x = low_res
        self.y = hight_res

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx], self.y[idx]


class SrClassifier_4090(Dataset):
    def __init__(self, low_res: np.ndarray, high_res: np.ndarray, labels: np.ndarray):
        self.low_res = low_res
        self.high_res = high_res
        self.label = labels

    def __len__(self):
        return self.low_res.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.low_res[idx], [
            self.high_res[idx],
            torch.LongTensor(self.label[idx]),
        ]
