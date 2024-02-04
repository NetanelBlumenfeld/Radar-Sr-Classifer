import datetime
import logging
import os
from typing import Optional

import numpy as np
import torch
from gestures.network.models.basic_model import BasicModel
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


def setup_logger(log_file_path):
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the logging level

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    return logger


def ensure_path_exists(path):
    """
    Checks if a given path exists, and if not, creates it.

    Parameters:
    path (str): The path to be checked and potentially created.

    Returns:
    None
    """
    if not os.path.exists(path):
        try:
            print(f"Creating directory at: {path}")  # Debugging statement
            os.makedirs(path)
        except PermissionError:
            print(f"Permission denied: Cannot create directory at '{path}'.")
        except Exception as e:
            print(f"An error occurred while creating directory: {e}")


def get_time_in_string():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H:%M:%S")


class CallbackProtocol:
    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        pass

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        pass

    def on_batch_begin(
        self, batch: Optional[int] = None, logs: Optional[dict] = None
    ) -> None:
        pass

    def on_batch_end(
        self, batch: Optional[int] = None, logs: Optional[dict] = None
    ) -> None:
        pass

    def on_eval_begin(self, logs: Optional[dict] = None) -> None:
        pass

    def on_eval_end(self, logs: Optional[dict] = None) -> None:
        pass


class CallbackHandler(CallbackProtocol):
    def __init__(self, callbacks: Optional[list[CallbackProtocol]] = None):
        self.callbacks = callbacks if callbacks is not None else []

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        for callback in self.callbacks:
            if hasattr(callback, "on_train_begin"):
                callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[dict] = None):
        for callback in self.callbacks:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, "on_epoch_begin"):
                callback.on_epoch_begin(epoch, **kwargs)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        for callback in self.callbacks:
            if hasattr(callback, "on_epoch_end"):
                callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: Optional[int] = None, logs: Optional[dict] = None):
        for callback in self.callbacks:
            if hasattr(callback, "on_batch_begin"):
                callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: Optional[int] = None, logs: Optional[dict] = None):
        for callback in self.callbacks:
            if hasattr(callback, "on_batch_end"):
                callback.on_batch_end(batch, logs)

    def on_eval_begin(self, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, "on_eval_begin"):
                callback.on_eval_begin(**kwargs)

    def on_eval_end(self, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, "on_eval_end"):
                callback.on_eval_end(**kwargs)


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
        self.best_model_path = ""

    def _add_cm(self, trues, preds, title: str):
        cm = confusion_matrix(np.concatenate(trues), np.concatenate(preds))
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
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

        # TODO - move to another place
        def _get_preds_for_best_models(model, loader):
            preds, trues = [], []
            for batch, labels in loader:
                batch, labels = model.reshape_to_model_output(
                    batch, labels, self.device
                )
                batch = batch
                labels[0] = labels[0]

                outputs = model(batch)
                pred_labels = outputs[1].cpu().detach().numpy().reshape(-1, 12)
                # pred_labels = outputs.cpu().detach().numpy().reshape(-1, 12)

                pred_labels = np.argmax(pred_labels, axis=1)
                preds.append(pred_labels)
                trues.append(labels[1].cpu().detach().numpy().reshape(-1))
                # trues.append(labels.cpu().detach().numpy().reshape(-1))

            return preds, trues

        if logs is None:
            return

        if self.with_cm:
            model = logs["model"].to(self.device)
            models = ["max_acc_model.pt", "min_loss_model.pt"]
            data_loader = logs["data_loader"]
            for model_name in models:
                model.load_state_dict(torch.load(self.best_model_path + model_name))
                preds, trues = _get_preds_for_best_models(model, data_loader)
                self._add_cm(preds, trues, model_name.split(".")[0])
        self.writer.close()

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        if logs is None:
            return
        metrics = logs["metrics"]
        for data in ["train", "val"]:
            for metric_name, metric_value in metrics[data].items():
                self.writer.add_scalar(f"{data}/{metric_name}", metric_value, epoch)


class SaveModel(CallbackProtocol):
    # TODO - save my metric name and operation
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
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "tracked_value": value,
            },
            path,
        )
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

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
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
                path = os.path.join(self.save_path + metric_tracker["metric_name"])
                self._save(
                    path=path,
                    model=logs["model"],
                    optimizer=logs["optimizer"],
                    epoch=epoch,
                    value=new_value,
                    value_name=value_name,
                )


class ProgressBar(CallbackProtocol):
    def __init__(
        self,
        loader_train,
        logger=None,
        training_desc: Optional[str] = None,
        verbose: int = 0,
        output_dir: str = "",
    ):
        self.loader_train = loader_train
        self.training_desc = training_desc
        self.verbose = verbose
        self.out_val = ""
        self.out_train = ""
        self.output_dir = output_dir
        self.logger = logger

    def _print_metrics(self, metrics: dict[str, dict]) -> str:
        res = ""
        for data in ["train", "val"]:
            for metric_name, metric_value in metrics[data].items():
                res += f"{metric_name}: {metric_value:.4f} "
        return res

    def _update_postfix_str(self):
        self.pbar.set_postfix_str(f"{self.out_train} {self.out_val}")

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        if self.verbose == 0:
            bar_format = "{l_bar}{bar}| [{n_fmt}/{total_fmt}]  {postfix}"
            self.pbar = tqdm(
                self.loader_train,
                total=len(self.loader_train),
                bar_format=bar_format,
                ncols=200,
            )
        elif self.verbose == 1 and self.logger is not None:
            self.logger.debug(f"Training - {self.training_desc}")

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        if self.verbose == 0:
            self.pbar.close()

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        print("\n")
        if self.verbose == 0:
            self.pbar.reset()
            self.pbar.set_description(f"Epoch {epoch}")
        elif self.verbose == 1 and self.logger is not None:
            self.logger.debug(f"Epoch {epoch} - {self.out_train} {self.out_val}")

        self.out_val = ""
        self.out_train = ""

    def on_batch_end(
        self, batch: Optional[int] = None, logs: Optional[dict] = None
    ) -> None:
        if logs is None:
            return
        metrics = logs["metrics"]
        self.out_train = f"Train - {self._print_metrics(metrics)}"
        if self.verbose == 0:
            self._update_postfix_str()
            self.pbar.update(1)

    def on_eval_end(self, logs: Optional[dict] = None) -> None:
        if logs is None:
            return
        metrics = logs["metrics"]
        self.out_val = f"Val - {self._print_metrics(metrics)}"
        if self.verbose == 0:
            self._update_postfix_str()


class Logger(CallbackProtocol):
    def __init__(self, base_dir: str):
        self.log_file = os.path.join(base_dir, "log.txt")
        self.base_dir = base_dir
        ensure_path_exists(base_dir)
        self.logger = setup_logger(self.log_file)

    def _print_metrics(self, metrics: dict[str, dict]) -> str:
        res = ""
        for metric_name, metric_value in metrics.items():
            res += f"{metric_name}: {metric_value:.4f} "
        return res

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        "log description about the data, model, training config, base path for outputs"
        if logs:
            txt1 = f" \n Training started at {get_time_in_string()}"
            txt2 = f"\n Base dir: {self.base_dir}"
            txt3 = f"\n train batches: {logs['data_info']['train']}, val batches: {logs['data_info']['val']}"
            txt = txt1 + txt2 + txt3
            self.logger.info(txt)

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        "log the best model path"
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        "log Epoch number, train and val metrics"
        if logs:
            lr = logs["train_info"]["lr"]
            print(str(lr) + "fd")
            txt1 = f"\n Epoch - {epoch}/{logs['train_info']['epochs']}, lr - {lr}"
            txt2 = f"\n Train - {self._print_metrics(logs['metrics']['train'])}"
            txt3 = f"\n Val - {self._print_metrics(logs['metrics']['val'])}"
            txt = txt1 + txt2 + txt3
            self.logger.info(txt)


if __name__ == "__main__":
    pass
