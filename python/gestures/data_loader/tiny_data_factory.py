import os
from functools import partial
from multiprocessing import Pool
from typing import Callable, Optional

import numpy as np
from gestures.data_loader.tiny_dataset import (
    ClassifierDataset,
    SrClassifier_4090,
    SrClassifierDataset_3080,
    SrDataSet_3080,
    SrDataSet_4090,
)
from gestures.data_loader.tiny_pipelines import (
    classifier_pipeline,
    sr_classifier_4090_pipeline,
    sr_classifier_time_4090_pipeline,
    sr_time_4090_pipeline,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

RANDOM_STATE = 42
TEST_SIZE = 0.2


def sr_classifier_loader(
    X: np.ndarray,
    labels: np.ndarray,
    norm_func: Callable,
    ds_func: Callable,
    time_domain: bool,
    processed_data: bool,
    use_pool: bool = True,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    data: (N,5,2,32,492)
    """
    data_set = SrClassifier_4090 if processed_data else SrClassifierDataset_3080
    if time_domain and processed_data:
        pipeline = partial(
            sr_classifier_time_4090_pipeline, norm_func=norm_func, ds_func=ds_func
        )

    elif not time_domain and processed_data:
        pipeline = partial(
            sr_classifier_4090_pipeline, norm_func=norm_func, ds_func=ds_func
        )
    else:
        raise ValueError("Not implemented")

    if use_pool:
        with Pool(os.cpu_count()) as p:
            result = p.map(pipeline, zip(X, labels))
        low_res = np.concatenate(list(map(lambda x: x[0], result)))
        high_res = np.concatenate(list(map(lambda x: x[1], result)))
        labels = np.concatenate(list(map(lambda x: x[2], result)))
    else:
        low_res, high_res, labels = pipeline((X, labels))
    del X

    (
        train_low_res,
        sub_split_low_res,
        train_high_res,
        sub_split_high_res,
        train_labels,
        sub_split_labels,
    ) = train_test_split(
        low_res,
        high_res,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_data_set = data_set(train_low_res, train_high_res, train_labels)
    del train_low_res, train_high_res, train_labels
    (
        val_low_res,
        test_low_res,
        val_high_res,
        test_high_res,
        val_labels,
        test_labels,
    ) = train_test_split(
        sub_split_low_res,
        sub_split_high_res,
        sub_split_labels,
        test_size=0.5,
        random_state=RANDOM_STATE,
    )
    test_data_set = data_set(test_low_res, test_high_res, test_labels)
    del test_low_res, test_high_res, test_labels
    val_data_set = data_set(val_low_res, val_high_res, val_labels)
    del val_low_res, val_high_res, val_labels
    return train_data_set, val_data_set, test_data_set


def sr_loader(
    X: np.ndarray,
    labels: np.ndarray,
    norm_func: Callable,
    ds_func: Callable,
    time_domain: bool,
    processed_data: bool,
    use_pool: bool = True,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    data: (N,5,2,32,492)
    """
    data_set = SrDataSet_4090 if processed_data else SrDataSet_3080
    if time_domain and processed_data:
        pipeline = partial(sr_time_4090_pipeline, norm_func=norm_func, ds_func=ds_func)

    else:
        raise ValueError("Not implemented")

    if use_pool:
        with Pool(os.cpu_count()) as p:
            result = p.map(pipeline, zip(X, labels))
        low_res = np.concatenate(list(map(lambda x: x[0], result)))
        high_res = np.concatenate(list(map(lambda x: x[1], result)))
    else:
        low_res, high_res = pipeline((X, labels))
    del X

    (
        train_low_res,
        sub_split_low_res,
        train_high_res,
        sub_split_high_res,
    ) = train_test_split(
        low_res,
        high_res,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_data_set = data_set(train_low_res, train_high_res)
    del train_low_res, train_high_res
    test_low_res, val_low_res, test_high_res, val_high_res = train_test_split(
        sub_split_low_res,
        sub_split_high_res,
        test_size=0.5,
        random_state=RANDOM_STATE,
    )
    test_data_set = data_set(test_low_res, test_high_res)
    del test_low_res, test_high_res
    val_data_set = data_set(val_low_res, val_high_res)
    return train_data_set, val_data_set, test_data_set


def classifier_loader(
    X: np.ndarray,
    labels: np.ndarray,
    norm_func: Callable,
    time_domain: bool,
    ds_func: Optional[Callable] = None,
    processed_data: Optional[Callable] = None,
    use_pool: bool = True,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    data: (N,5,2,32,492)
    """
    pipeline = partial(
        classifier_pipeline,
        norm_func=norm_func,
        ds_func=ds_func,
        time_domain=time_domain,
    )
    if use_pool:
        with Pool(os.cpu_count()) as p:
            result = p.map(pipeline, zip(X, labels))
        x = np.concatenate(list(map(lambda x: x[0], result)))
        labels = np.concatenate(list(map(lambda x: x[1], result)))
    else:
        x, labels = pipeline((X, labels))
    del X

    (
        x_train,
        x_split,
        train_labels,
        split_labels,
    ) = train_test_split(
        x,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_data_set = ClassifierDataset(x_train, train_labels)
    del x_train, train_labels
    x_val, x_test, val_labels, test_labels = train_test_split(
        x_split,
        split_labels,
        test_size=0.5,
        random_state=RANDOM_STATE,
    )
    val_data_set = ClassifierDataset(x_val, val_labels)
    test_data_set = ClassifierDataset(x_test, test_labels)
    return train_data_set, val_data_set, test_data_set
