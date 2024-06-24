import gc
import os

import torch
import torch.optim.lr_scheduler as lr_scheduler
from gestures.network.callbacks.callback_handler import CallbackHandler
from gestures.network.metric.metric_tracker import (
    AccMetricTrackerSrClassifier,
    LossMetricTrackerSrClassifier,
)
from gestures.network.models.basic_model import BasicModel
from torch.utils.data.dataloader import DataLoader


def train(model, loader_train, device, optimizer, loss_metric, acc_metric):
    model.train()
    for batch, labels in loader_train:
        batch, labels = model.reshape_to_model_output(batch, labels, device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = loss_metric.update(outputs, labels)
        if torch.isnan(loss).any():
            raise ValueError("Loss is None")
        loss.backward()  # type: ignore
        optimizer.step()
        acc_metric.update(outputs, labels)


def validate(model, dataset: DataLoader, device, loss_metric, acc_metric):
    model.eval()
    with torch.no_grad():
        for batch, labels in dataset:
            batch, labels = model.reshape_to_model_output(batch, labels, device)
            outputs = model(batch)
            _ = loss_metric.update(outputs, labels)
            acc_metric.update(outputs, labels)
            del outputs, batch, labels
            gc.collect()
            torch.cuda.empty_cache()


def get_pred_true_labels(task, acc_metric):
    if task == "classifier":
        true = acc_metric.metrics["ClassifierAccuracy"].true_labels
        pred = acc_metric.metrics["ClassifierAccuracy"].pred_labels
    elif task == "sr_classifier":
        true = acc_metric.classifier_acc.metrics["ClassifierAccuracy"].true_labels
        pred = acc_metric.classifier_acc.metrics["ClassifierAccuracy"].pred_labels
    else:
        raise ValueError(f"Unknown task: {task}")
    return true, pred


class Runner:
    def __init__(
        self,
        model: torch.nn.Module,
        loader_train: DataLoader,
        loader_validation: DataLoader,
        loader_test: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,  # type: ignore
        loss_metric: LossMetricTrackerSrClassifier,
        acc_metric: AccMetricTrackerSrClassifier,
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
            "true_labels": None,
            "pred_labels": None,
            "model_name": None,
        }

        self.lr_s = lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)

    def reset(self):
        self.acc_metric.reset()
        self.loss_metric.reset()

    def run(self, epochs: int):
        self.logs["train_info"]["epochs"] = epochs

        self.callbacks.on_train_begin(self.logs)
        for i in range(epochs):
            self.callbacks.on_epoch_begin(i)
            train(
                self.model,
                self.loader_train,
                self.device,
                self.optimizer,
                self.loss_metric,
                self.acc_metric,
            )
            self.logs["metrics"]["train"] = (
                self.acc_metric.value | self.loss_metric.value
            )

            self.reset()
            validate(
                self.model,
                self.loader_validation,
                self.device,
                self.loss_metric,
                self.acc_metric,
            )
            self.logs["metrics"]["val"] = self.acc_metric.value | self.loss_metric.value

            self.reset()
            self.callbacks.on_epoch_end(i, self.logs)
            self.lr_s.step()
            self.logs["train_info"]["lr"] = self.optimizer.param_groups[0]["lr"]

        self.test_evaluation()
        self.callbacks.on_train_end(self.logs)

    def test_evaluation(self):
        best_models_dir = os.path.join(self.base_dir, "model")
        for file_name in os.listdir(best_models_dir):
            if file_name.endswith(".pth"):
                self.reset()
                model_path = os.path.join(best_models_dir, file_name)
                model, _, _, _ = BasicModel.load_pre_train_model(
                    self.device, model_path
                )
                model_name = f'{file_name.split(".")[0]}_{model.model_name}'
                validate(
                    model,
                    self.loader_test,
                    self.device,
                    self.loss_metric,
                    self.acc_metric,
                )
                self.logs["metrics"]["test"] = (
                    self.acc_metric.value | self.loss_metric.value
                )
                self.logs["true_labels"], self.logs["pred_labels"] = (
                    get_pred_true_labels(self.task, self.acc_metric)
                )

                self.logs["model_name"] = model_name
                self.callbacks.on_eval_end(self.logs)
