import os

import numpy as np
import torch
from gestures.network.callbacks.callback_handler import CallbackProtocol
from gestures.utils_os import ensure_path_exists
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.tensorboard.writer import SummaryWriter


class BaseTensorBoardTracker(CallbackProtocol):
    def __init__(
        self,
        base_dir: str,
        classes_name: list[str],
        with_cm: bool = True,
    ):
        board_dir = os.path.join(base_dir, "tensorboard")
        ensure_path_exists(board_dir)
        self.writer = SummaryWriter(log_dir=board_dir)
        self.classes_name = classes_name
        self.with_cm = with_cm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_model_path = os.path.join(base_dir, "model")

    def _add_cm(self, trues, preds, title: str):
        cm = confusion_matrix(trues, preds)
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.divide(
            cm.astype("float"),
            cm_sum,
            out=np.zeros_like(cm, dtype=float),
            where=cm_sum != 0,
        )
        # cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        # Create the ConfusionMatrixDisplay instance
        fig, ax = plt.subplots(figsize=(20, 20))
        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=cm_normalized,
            display_labels=self.classes_name,
        )

        # Plot with percentages
        cm_display.plot(cmap="Blues", values_format=".2%", ax=ax)
        self.writer.add_figure(f"confusion_matrix/{title}", cm_display.figure_, 0)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is None:
            return
        self._add_metrics_scalars(logs["metrics"], ["train", "val"], epoch)

    def on_eval_end(self, logs: dict | None = None) -> None:
        if logs is None:
            return
        self._add_metrics_scalars(logs["metrics"], ["test"], 0)
        preds_labels = logs["pred_labels"]
        true_labels = logs["true_labels"]
        self._add_confusion_matrix(
            preds_labels, true_labels, "test_data", logs["model_name"]
        )

    def _add_metrics_scalars(self, metrics: dict, data_kind, epoch: int):
        for data_kind in data_kind:
            for metric_name, metric_value in metrics[data_kind].items():
                self.writer.add_scalar(
                    f"{data_kind}/{metric_name}", metric_value, epoch
                )

    def _add_confusion_matrix(self, true_labels, pred_labels, data_name, model_name):
        preds = np.concatenate(pred_labels, axis=0)
        trues = np.concatenate(true_labels, axis=0)
        matrix_name = f"{data_name}_{model_name}"
        self._add_cm(preds, trues, matrix_name)
        self.writer.close()
