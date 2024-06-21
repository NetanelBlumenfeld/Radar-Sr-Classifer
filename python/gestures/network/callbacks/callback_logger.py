import datetime
import logging
import os
from typing import Optional

from gestures.network.callbacks.callback_handler import CallbackProtocol
from gestures.utils_os import ensure_path_exists


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


def get_time_in_string():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H:%M:%S")


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
        if logs:
            txt1 = "\nTest result"
            txt2 = (
                f"\n Test - {[self._print_metrics(f) for f in logs['metrics']['test']]}"
            )
            txt = txt1 + txt2
            self.logger.info(txt)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        "log Epoch number, train and val metrics"
        if logs:
            lr = logs["train_info"]["lr"]
            txt1 = f"\n Epoch - {epoch}/{logs['train_info']['epochs']}, lr - {lr}"
            txt2 = f"\n Train - {self._print_metrics(logs['metrics']['train'])}"
            txt3 = f"\n Val - {self._print_metrics(logs['metrics']['val'])}"
            txt = txt1 + txt2 + txt3
            self.logger.info(txt)
