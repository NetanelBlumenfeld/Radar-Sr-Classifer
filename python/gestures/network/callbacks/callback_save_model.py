import os

import torch
from gestures.network.callbacks.callback_handler import CallbackProtocol
from gestures.network.models.basic_model import BasicModel
from gestures.utils_os import ensure_path_exists


class SaveModel(CallbackProtocol):
    def __init__(
        self,
        base_dir: str,
        metrics: list[str],
        opts: list[str],
        save_best: bool = True,
    ):
        self.save_path = os.path.join(base_dir, "model")
        self.best_model_path = None
        self.save_best = save_best
        assert len(metrics) == len(opts)
        self.metric_trackers = self._set_metric_trackers(metrics, opts)

        ensure_path_exists(self.save_path)

    @staticmethod
    def _parse_metric_name(metric: str) -> tuple:
        """
        Parse the metric name to extract the data name and metric name.

        Args:
        metric (str): The metric identifier.

        Returns:
        tuple: A tuple of (data_name, metric_name).
        """
        parts = metric.split("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid metric format: {metric}")
        return tuple(parts)

    def _save(
        self,
        path: str,
        model: BasicModel,
        optimizer,
        epoch: int,
        value: float,
        value_name: str,
    ):
        state = {
            "model": model,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "tracked_value": value,
        }
        torch.save(state, path)
        print(f"\nSaved model at - {path}, {value_name} - {value}")

    def _set_metric_trackers(self, metrics: list[str], opts: list[str]) -> list[dict]:
        """
        Initialize metric trackers based on provided metrics and options.

        Args:
        metrics (list[str]): List of metric identifiers.
        opts (list[str]): List of options ('max' or 'min').

        Returns:
        list: List of initialized metric trackers.
        """
        trackers = []
        for metric, opt in zip(metrics, opts):
            data_name, metric_name = self._parse_metric_name(metric)
            operation = max if opt == "max" else min
            value = -torch.inf if opt == "max" else torch.inf
            tracker = {
                "data_name": data_name,
                "metric_name": metric_name,
                "opt": opt,
                "value": value,
                "operation": operation,
            }
            trackers.append(tracker)
        return trackers

    def _construct_filename(self, tracker, epoch):
        """
        Construct the filename for saving the model.

        Args:
        tracker (dict): The metric tracker dictionary.
        epoch (int): The current epoch number.

        Returns:
        str: The constructed filename.
        """
        filename = f"{tracker['data_name']}_{tracker['metric_name']}"
        if not self.save_best:
            filename += f"_epoch{epoch}"
        return filename + ".pt"

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is None:
            raise ValueError("logs is None on ModelSave callback")
        for metric_tracker in self.metric_trackers:
            new_value = logs["metrics"][metric_tracker["data_name"]][
                metric_tracker["metric_name"]
            ]
            prev_value = metric_tracker["value"]
            if metric_tracker["operation"](new_value, prev_value) == new_value:
                value_name = self._construct_filename(metric_tracker, epoch)
                metric_tracker["value"] = new_value
                path = os.path.join(
                    self.save_path, metric_tracker["metric_name"] + ".pth"
                )
                self._save(
                    path=path,
                    model=logs["model"],
                    optimizer=logs["optimizer"],
                    epoch=epoch,
                    value=new_value,
                    value_name=value_name,
                )
