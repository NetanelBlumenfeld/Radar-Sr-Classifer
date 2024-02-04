import os
from typing import Optional

import torch as torch
from gestures.network.experiment_tracker import (
    BaseTensorBoardTracker,
    CallbackHandler,
    ProgressBar,
    SaveModel,
    get_time_in_string,
)
from gestures.network.metric.accuracy import acc_srcnn_tiny_radar
from gestures.network.metric.loss import LossFunctionSRTinyRadarNN, LossType, SimpleLoss
from gestures.network.metric.metric_tracker import (
    AccuracyMetric,
    LossMetric,
    LossMetricSRTinyRadarNN,
)
from gestures.network.models.classifiers.tiny_radar import TinyRadarNN
from gestures.network.models.sr_classifier.SRCnnTinyRadar import (
    CombinedSRCNNClassifier,
    CombinedSRDrlnClassifier,
)
from gestures.network.models.super_resolution.drln import Drln
from gestures.network.runner import Runner
from torch.utils.data import DataLoader


def train_drln_tiny_radar(
    training_generator: DataLoader,
    val_generator: DataLoader,
    output_dir: str,
    gestures: list[str],
    device: torch.device,
    epochs: int,
    batch_size: int,
    tiny_checkpoint: Optional[str] = None,
    drln_checkpoint: Optional[str] = None,
    verbose: int = 1,
):
    # Training parameters
    lr = 0.0015
    w_sr = 1
    w_c = 1
    loss_type_sr = LossType.MSSSIML1
    froze_sr = False
    froze_classifier = False

    # models setup
    if tiny_checkpoint:
        classifier, _, _, _ = TinyRadarNN.load_model(
            model_dir=tiny_checkpoint,
            optimizer_class=torch.optim.Adam,
            optimizer_args=lr,
            device=device,
        )
    else:
        classifier = TinyRadarNN(numberOfGestures=len(gestures)).to(device)
    if drln_checkpoint:
        sr, _, _, _ = Drln.load_model(
            model_dir=drln_checkpoint,
            optimizer_class=torch.optim.Adam,
            optimizer_args={"lr": lr},
            device=device,
            **{"num_drln_blocks": 2, "scale": 4, "num_channels": 2},
        )
    else:
        sr = Drln(num_drln_blocks=2, scale=4, num_channels=2).to(device)
    if froze_sr:
        sr.freeze_weights()
    if froze_classifier:
        classifier.freeze_weights()
    model = CombinedSRDrlnClassifier(sr, classifier, scale_factor=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    loss_func = LossFunctionSRTinyRadarNN(
        loss_type_srcnn=loss_type_sr,
        loss_type_classifier=LossType.CrossEntropy,
        wight_srcnn=w_sr,
        wight_classifier=w_c,
    )
    loss_metric = LossMetricSRTinyRadarNN(metric_function=loss_func)
    acc_metric = AccuracyMetric(metric_function=acc_srcnn_tiny_radar)

    # paths
    train_config = (
        f"lr_{lr}_batch_size_{batch_size}_{loss_metric.name}_w_sr_{w_sr}_w_c_{w_c}"
    )
    if not tiny_checkpoint and drln_checkpoint:
        task_type = "sr_ck_classifier"
    elif not drln_checkpoint and tiny_checkpoint:
        task_type = "sr_classifier_ck"
    elif drln_checkpoint and tiny_checkpoint:
        task_type = "sr_ck_classifier_ck"
    else:
        task_type = "sr_classifier"
    experiment_name = os.path.join(
        task_type,  # model type
        model.model_name,  # model name
        train_config,  # training configuration
        get_time_in_string(),
    )
    print(f"experiment name - {experiment_name}")
    t_board_dir = output_dir + "tensorboard/" + experiment_name
    save_model_dir = output_dir + "models/" + experiment_name

    # callbacks
    t_board = BaseTensorBoardTracker(
        base_dir=t_board_dir,
        classes_name=gestures,
        # best_model_path=save_model_dir,
    )
    saver = SaveModel(save_model_dir)
    prog_bar = ProgressBar(
        training_generator,
        training_desc=experiment_name,
        verbose=verbose,
    )
    callbacks = CallbackHandler([t_board, saver, prog_bar])

    torch.cuda.empty_cache()

    runner = Runner(
        model,
        training_generator,
        val_generator,
        device,
        optimizer,
        loss_metric,
        acc_metric,
        callbacks,
    )
    runner.run(epochs)


def train_drln(
    training_generator: DataLoader,
    val_generator: DataLoader,
    output_dir: str,
    gestures: list[str],
    device: torch.device,
    epochs: int,
    batch_size: int,
    verbose: int = 0,
    checkpoint: Optional[str] = None,
):
    lr = 0.0018

    model = Drln(num_drln_blocks=2, num_channels=2).to(device)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))

        print(f"loaded model from {checkpoint}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    loss_criterion = SimpleLoss(loss_function=LossType.MSSSIML1)
    loss_metric = LossMetric(metric_function=loss_criterion, kind="loss")
    acc_metric = LossMetric(metric_function=loss_criterion, kind="acc")

    # paths
    train_config = f"lr_{lr}_batch_size_{batch_size}_{loss_metric.name}"
    experiment_name = os.path.join(
        "sr",  # model type
        model.model_name,  # model name
        train_config,  # training configuration
        get_time_in_string(),
    )
    t_board_dir = output_dir + "tensorboard/" + experiment_name
    save_model_dir = output_dir + "models/" + experiment_name

    print(f"save dir - {save_model_dir}")
    print(f"t_board_dir - {t_board_dir}")

    # callbacks
    t_board = BaseTensorBoardTracker(
        base_dir=t_board_dir,
        classes_name=gestures,
        with_cm=False,
    )
    saver = SaveModel(save_model_dir)
    prog_bar = ProgressBar(
        training_generator, training_desc=experiment_name, verbose=verbose
    )
    callbacks = CallbackHandler([t_board, saver, prog_bar])
    torch.cuda.empty_cache()

    runner = Runner(
        model,
        training_generator,
        val_generator,
        device,
        optimizer,
        loss_metric,
        acc_metric,
        callbacks,
    )
    runner.run(epochs)
