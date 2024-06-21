import torch
from gestures.network.metric.criterion_factory import CriterionFactory, MetricCriterion
from gestures.network.metric.custom_criterion import AccuracyMetric

EPSILON = 1e-6


class BasicMetricTracker:
    def __init__(self, metric: MetricCriterion, wight: float = 1.0):
        self.metric_func = CriterionFactory.get_loss_function(metric.name)
        self.wight = wight
        self.name = metric.name
        self.running_total = 0
        self.values = []

    @property
    def value(self) -> float:
        res = sum(self.values) / (self.running_total + EPSILON)
        return res

    def reset(self):
        self.values = []
        self.running_total = 0

    def update(self, preds: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        self.running_total += 1
        res = self.metric_func(preds, true)
        self.values.append(res.item())
        return res


class LossMetricTracker:
    def __init__(self, loss_metrics: list[dict]):
        self.metrics = [BasicMetricTracker(**metric) for metric in loss_metrics]
        self._set_name()
        self.total_loss = []
        self.running_total = 0

    @property
    def value(self) -> dict[str, float]:
        res = {"total_loss": sum(self.total_loss) / (self.running_total + EPSILON)}
        for metric in self.metrics:
            res[f"loss_{metric.name}"] = metric.value
        return res

    def _set_name(self):
        self.name = "loss"
        for m in self.metrics:
            self.name += "_" + m.name

    def reset(self):
        self.running_total = 0
        self.total_loss = []
        for metric in self.metrics:
            metric.reset()

    def update(self, preds: torch.Tensor, true: torch.Tensor):
        loss = 0
        self.running_total += 1
        for metric in self.metrics:
            sub_loss = metric.update(preds, true)
            loss += sub_loss * metric.wight
        self.total_loss.append(loss.item())
        return loss


class AccMetricTracker:
    def __init__(self, acc_metrics: list[MetricCriterion]):
        self.metrics = []
        for metric in acc_metrics:
            if metric.name == "ClassifierAccuracy":
                self.metrics.append(AccuracyMetric())
            else:
                self.metrics.append(BasicMetricTracker(metric))

    @property
    def value(self) -> dict[str, float]:
        res = {}
        for metric in self.metrics:
            res[f"acc_{metric.name}"] = metric.value
        return res

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update(self, preds: torch.Tensor, true: torch.Tensor):
        for metric in self.metrics:
            _ = metric.update(preds, true)


class LossMetricTrackerSrClassifier:
    def __init__(
        self,
        classifier_tracker: LossMetricTracker,
        sr_tracker: LossMetricTracker,
        sr_weight: float,
        classifier_weight: float,
    ):
        self.classifier_tracker = classifier_tracker
        self.sr_tracker = sr_tracker
        self.sr_weight = sr_weight
        self.classifier_weight = classifier_weight
        self.running_total = 0
        self._set_name()

        self.sr_loss = []
        self.classifier_loss = []
        self.total_loss = []

    @property
    def value(self) -> dict[str, float]:
        sr_dict = {f"sr_{key}": value for key, value in self.sr_tracker.value.items()}
        classifier_dict = {
            f"classifier_{key}": value
            for key, value in self.classifier_tracker.value.items()
        }
        total_loss = {
            "total_loss": sum(self.total_loss) / (self.running_total + EPSILON)
        }
        return total_loss | sr_dict | classifier_dict

    def _set_name(self):
        self.name = (
            f"sr_{self.sr_tracker.name}_classifier_{self.classifier_tracker.name}"
        )

    def reset(self):
        self.sr_tracker.reset()
        self.classifier_tracker.reset()
        self.running_total = 0
        self.sr_loss = []
        self.classifier_loss = []
        self.total_loss = []

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        self.running_total += 1
        pred_sr, true_sr = outputs[0], labels[0]
        pred_classifier, true_classifier = outputs[1], labels[1]
        loss_sr = self.sr_tracker.update(pred_sr, true_sr)
        loss_classifier = self.classifier_tracker.update(
            pred_classifier, true_classifier
        )
        self.sr_loss.append(loss_sr.item())
        self.classifier_loss.append(loss_classifier.item())
        loss = loss_sr * self.sr_weight + loss_classifier * self.classifier_weight
        self.total_loss.append(loss)
        return loss


class AccMetricTrackerSrClassifier:
    def __init__(self, sr_acc: AccMetricTracker, classifier_acc: AccMetricTracker):
        self.sr_acc = sr_acc
        self.classifier_acc = classifier_acc

    @property
    def value(self):
        sr_dict = {f"sr_{key}": value for key, value in self.sr_acc.value.items()}
        classifier_dict = {
            f"classifier_{key}": value
            for key, value in self.classifier_acc.value.items()
        }
        return sr_dict | classifier_dict

    def reset(self):
        self.sr_acc.reset()
        self.classifier_acc.reset()

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        pred_sr, true_sr = outputs[0], labels[0]
        pred_classifier, true_classifier = outputs[1], labels[1]
        self.sr_acc.update(pred_sr, true_sr)
        self.classifier_acc.update(pred_classifier, true_classifier)
