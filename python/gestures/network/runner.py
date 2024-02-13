import gc
import os
from typing import Any

import torch
import torch.optim.lr_scheduler as lr_scheduler
from gestures.network.experiment_tracker import CallbackHandler
from gestures.network.metric.metric_tracker import MetricTracker
from torch.utils.data.dataloader import DataLoader


def get_best_model_paths(base_dir):
    best_model_dir = os.path.join(base_dir, "model")
    res = []
    for file_name in os.listdir(best_model_dir):
        if file_name.endswith("pth"):
            res.append(best_model_dir + "/" + file_name)
    return res


def get_models(base_dir, device, model_cls):
    models = []
    models_path = get_best_model_paths(base_dir)
    for p in models_path:
        model, _, _, _ = model_cls.load_model(device, p)
        models.append(model)
    return models


class Runner:
    def __init__(
        self,
        model: torch.nn.Module,
        loader_train: DataLoader,
        loader_validation: DataLoader,
        loader_test: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss_metric: MetricTracker,
        acc_metric: MetricTracker,
        callbacks: CallbackHandler,
        base_dir: str,
        task: str,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loader_train = loader_train
        self.loader_validation = loader_validation
        self.loader_test = loader_test
        self.acc_metric = acc_metric
        self.loss_metric = loss_metric
        self.callbacks = callbacks
        self.base_dir = base_dir
        self.task = task
        self.logs = {
            "model": self.model,
            "optimizer": self.optimizer,
            "metrics": {"train": None, "val": None, "test": None},
            "data_test": self.loader_test,
            "data_validation": self.loader_validation,
            "data_info": {
                "train": len(self.loader_train),
                "val": len(self.loader_validation),
            },
            "train_info": {
                "epochs": None,
                "lr": self.optimizer.param_groups[0]["lr"],
            },
            "task": self.task,
        }

        self.lr_s = lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)

    def reset(self):
        self.acc_metric.reset()
        self.loss_metric.reset()

    def run(self, epochs: int):
        self.logs["train_info"]["epochs"] = epochs

        self.callbacks.on_train_begin(self.logs)
        for i in range(epochs):
            print(i)
            self.callbacks.on_epoch_begin(i)
            self.logs["metrics"]["train"] = self.train()
            self.reset()
            self.logs["metrics"]["val"] = self.validate("val", self.loader_validation)
            self.reset()
            self.callbacks.on_epoch_end(i, self.logs)
            self.lr_s.step()
            self.logs["train_info"]["lr"] = self.optimizer.param_groups[0]["lr"]

        self.reset()
        self.logs["metrics"]["test"] = self.test_evaluation()
        self.callbacks.on_train_end(self.logs)

    def train(self) -> dict[str, Any]:
        self.model.train()

        for batch, labels in self.loader_train:
            batch, labels = self.model.reshape_to_model_output(
                batch, labels, self.device
            )
            self.callbacks.on_batch_begin(logs=self.logs)
            self.logs["metrics"]["train"] = self.train_batch(batch, labels)
            del batch, labels
            self.callbacks.on_batch_end(logs=self.logs)
        return self.logs["metrics"]["train"]

    def train_batch(self, batch, labels) -> dict[str, Any]:
        self.optimizer.zero_grad()
        outputs = self.model(batch)
        loss = self.loss_metric.update(outputs, labels)
        if torch.isnan(loss).any():
            raise ValueError("Loss is None")
        loss.backward()  # type: ignore
        self.optimizer.step()
        self.acc_metric.update(outputs, labels)
        return self.loss_metric.value | self.acc_metric.value

    def validate(self, kind: str, dataset: DataLoader, model=None):
        if model is not None:
            self.model = model
        self.model.eval()
        with torch.no_grad():
            for batch, labels in dataset:
                batch, labels = self.model.reshape_to_model_output(
                    batch, labels, self.device
                )
                self.callbacks.on_eval_begin()
                self.logs["metrics"][kind] = self.validate_batch(batch, labels)
                del batch, labels
                self.callbacks.on_eval_end(logs=self.logs)
        return self.logs["metrics"][kind]

    def validate_batch(self, batch, labels) -> dict[str, Any]:

        outputs = self.model(batch)
        _ = self.loss_metric.update(outputs, labels)
        self.acc_metric.update(outputs, labels)
        del outputs, batch, labels
        gc.collect()
        torch.cuda.empty_cache()

        return self.loss_metric.value | self.acc_metric.value

    def test_evaluation(self):
        models = get_models(self.base_dir, self.device, self.model)
        res = []
        for m in models:
            res.append(self.validate("test", self.loader_test, m))
        return res
