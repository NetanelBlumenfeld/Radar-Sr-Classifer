import os
from typing import Optional

import numpy as np
import torch
from gestures.network.callbacks.callback_handler import CallbackProtocol
from gestures.utils_os import ensure_path_exists
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter


class BaseTensorBoardTracker(CallbackProtocol):
    def __init__(
        self,
        base_dir: str,
        classes_name: list[str],
        # best_model_path: str,
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

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        """loading the best model and calculate the confusion matrix"""

        def _get_preds_for_best_models(model, loader: DataLoader, task: str):
            preds_list, trues_list = [], []
            for batch, labels in loader:
                batch, labels = model.reshape_to_model_output(
                    batch, labels, self.device
                )
                batch = batch

                outputs = model(batch)
                preds = outputs[1] if task == "sr_classifier" else outputs
                class_labels = labels[1] if task == "sr_classifier" else labels

                pred_labels = preds.cpu().detach().numpy().reshape(-1, 12)

                pred_labels = np.argmax(pred_labels, axis=1)
                preds_list.append(pred_labels)
                trues_list.append(class_labels.cpu().detach().numpy().reshape(-1))
                # trues.append(labels.cpu().detach().numpy().reshape(-1))

            return preds_list, trues_list

        if logs is None:
            return

        if self.with_cm:
            model_cls = logs["model"]
            task = logs["task"]
            for file_name in os.listdir(self.best_model_path):
                if file_name.endswith(".pth"):
                    model_path = os.path.join(self.best_model_path, file_name)
                    model, _, _, _ = model_cls.load_model(self.device, model_path)
                    for data_name in ["data_test", "data_validation"]:
                        data_loader = logs[data_name]
                        if task == "classifier" or task == "sr_classifier":
                            preds, trues = _get_preds_for_best_models(
                                model, data_loader, task
                            )
                            preds = np.concatenate(preds, axis=0)
                            trues = np.concatenate(trues, axis=0)
                            matrix_name = f"{data_name}_{file_name.split('.')[0]}"
                            self._add_cm(preds, trues, matrix_name)

        self.writer.close()

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        if logs is None:
            return
        metrics = logs["metrics"]
        for data in ["train", "val"]:
            for metric_name, metric_value in metrics[data].items():
                self.writer.add_scalar(f"{data}/{metric_name}", metric_value, epoch)
