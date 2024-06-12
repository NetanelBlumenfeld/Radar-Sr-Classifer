import torch
from torch.utils.data import Dataset


class ClassifierDataset(Dataset):
    def __init__(self, dataX, dataY, pre_train_process=None, on_train_process=None):
        # shape (N, 5, 2 ,32,492)
        assert dataX.shape[1:] == (5, 2, 32, 492)
        if pre_train_process:
            dataX = pre_train_process(dataX)
        self.x_train = dataX
        self.label = dataY
        self.on_train_process = on_train_process

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.x_train[idx]
        if self.on_train_process:
            data = self.on_train_process(self.x_train[idx])

        return data, torch.LongTensor(self.label[idx])


class SrDataset(Dataset):
    def __init__(self, dataX, pre_train_process=None, hr_to_lr=None):
        # shape (N, 5, 2 ,2,32,492)
        if pre_train_process:
            self.y = pre_train_process(dataX)
        if hr_to_lr:
            self.x = hr_to_lr(dataX)
        assert self.y is not None
        assert self.x.shape[1:4] == (5, 2, 2)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        low_res = self.x[idx]
        d0, d1, d2, d3, d4 = low_res.shape
        low_res = low_res.reshape(d0 * d1, d2, d3, d4)
        high_res = self.y[idx]
        d0, d1, d2, d3, d4 = high_res.shape
        high_res = high_res.reshape(d0 * d1, d2, d3, d4)

        return low_res, high_res


class SrClassifierDataset(Dataset):
    def __init__(
        self, dataX, dataY, pre_train_process=None, hr_to_lr=None, on_train_process=None
    ):
        # shape (N, 5, 2 ,2,32,492)
        if pre_train_process:
            self.y = pre_train_process(dataX)
        if hr_to_lr:
            self.x = hr_to_lr(dataX)  # low res
        assert self.y is not None
        assert self.x.shape[1:4] == (5, 2, 2)
        self.label = dataY
        self.on_train_process = on_train_process

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        low_res = self.x[idx]
        d0, d1, d2, d3, d4 = low_res.shape
        # low_res = low_res.reshape(d0 * d1, d2, d3, d4)
        high_res = self.y[idx]
        d0, d1, d2, d3, d4 = high_res.shape
        # high_res = high_res.reshape(d0 * d1, d2, d3, d4)
        if self.on_train_process:
            low_res = self.on_train_process(low_res)
            high_res = self.on_train_process(high_res)

        return low_res, [high_res, torch.LongTensor(self.label[idx])]
