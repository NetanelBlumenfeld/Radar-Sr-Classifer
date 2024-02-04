from dataclasses import dataclass, field

import torch
from gestures.network.metric.loss import LossFactory, LossType


class MetricTracker:
    EPSILON = 1e-5

    def __init__(
        self, kind: str, metrics_names: list[LossType], metric_wights: list[float]
    ):
        assert len(metrics_names) == len(metric_wights)
        assert sum(metric_wights) == 1
        self.kind = kind
        self.name = kind + "_" + self._get_metric_name(metrics_names, metric_wights)
        self.dict_tracker = self._get_dict_tracker(metrics_names, metric_wights)
        self.running_total = 0
        self.loss = []

    @property
    def value(self) -> dict[str, float]:
        res = {self.kind: sum(self.loss) / (self.running_total + self.EPSILON)}
        for metric_name, metric in self.dict_tracker.items():
            res[metric_name] = sum(metric["values"]) / (
                self.running_total + self.EPSILON
            )

        return res

    def _get_metric_name(
        self, metrics_names: list[LossType], metric_wights: list[float]
    ) -> str:
        metric_name = ""
        for metric, w in zip(metrics_names, metric_wights):
            metric_name += f"{metric.name}_{w}_"
        return metric_name[:-1]

    def _get_dict_tracker(
        self, metrics_names: list[LossType], metric_wights: list[float]
    ) -> dict:
        res = {}
        for metric, w in zip(metrics_names, metric_wights):
            metric_func = LossFactory.get_loss_function(metric.name)
            res[self.kind + "_" + metric.name] = {
                "wight": w,
                "values": [],
                "metric": metric_func,
            }
        return res

    def reset(self):
        self.running_total = 0
        self.loss = []
        for metric in self.dict_tracker.values():
            metric["values"] = []

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        self.running_total += 1
        loss = 0
        for metric in self.dict_tracker.values():
            sub_loss = metric["metric"](outputs, labels)

            loss += sub_loss * metric["wight"]

            metric["values"].append(sub_loss.item())
        self.loss.append(loss.item())

        if self.kind == "loss":
            return loss


class LossMetric:
    def __init__(self, metric_function, kind):
        self.metric_function = metric_function
        self.name = kind + "_" + metric_function.name
        self.running_total = 0
        self.values = []
        self.kind = kind

    @property
    def value(self) -> dict[str, float]:
        return {self.kind: sum(self.values) / self.running_total}

    def reset(self):
        self.values = []
        self.running_total = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        self.running_total += 1
        loss = self.metric_function(outputs, labels)
        self.values.append(loss.item())
        if self.kind == "loss":
            return loss


class LossMetricSRTinyRadarNN:
    def __init__(self, metric_function):
        self.metric_function = metric_function
        self.name = f"loss_{metric_function.name}"

        self.loss_srcnn_list = []
        self.loss_classifier_list = []
        self.values = []
        self.running_total = 0

    @property
    def value(self) -> dict[str, float]:
        return {
            "loss": sum(self.values) / (self.running_total + 1e-5),
            "loss_srcnn": sum(self.loss_srcnn_list) / (self.running_total + 1e-5),
            "loss_classifier": sum(self.loss_classifier_list)
            / (self.running_total + 1e-5),
        }

    def reset(self):
        self.values = []
        self.loss_classifier_list = []
        self.loss_srcnn_list = []
        self.running_total = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        self.running_total += 1
        loss, loss_classifier, loss_srcnn = self.metric_function(outputs, labels)
        self.values.append(loss.item())
        self.loss_srcnn_list.append(loss_srcnn.item())
        self.loss_classifier_list.append(loss_classifier.item())
        return loss


@dataclass
class AccuracyMetric:
    values: list[float] = field(default_factory=list)
    running_total: float = 0.0
    metric_function: callable = field(default_factory=None)

    @property
    def value(self):
        return {"acc": 100 * (sum(self.values) / self.running_total)}

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        total, correct = self.metric_function(outputs, labels)
        self.values.append(correct)
        self.running_total += total

    def reset(self):
        self.values = []
        self.running_total = 0.0
