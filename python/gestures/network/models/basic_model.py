import torch
import torch.nn as nn


class BasicModel(nn.Module):
    def __init__(self, model_name: str):
        super(BasicModel, self).__init__()
        self.model_name = model_name

    @classmethod
    def load_model(
        cls,
        device: torch.device,
        model_dir: str,
        optimizer_class=None,
        optimizer_args=None,
    ):
        # Create an instance of the subclass with provided arguments

        # Load the checkpoint
        checkpoint = torch.load(model_dir, map_location=device)
        model = checkpoint["model"].to(device)

        # Initialize the optimizer
        optimizer = None
        if optimizer_class is not None:
            optimizer = optimizer_class(model.parameters(), **optimizer_args)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Other information from checkpoint
        epoch = checkpoint["epoch"]
        loss = checkpoint["tracked_value"]
        return model, optimizer, epoch, loss

    def freeze_weights(self):
        # Freeze all weights in the model
        for param in self.parameters():
            param.requires_grad = False
        self.model_name += "_frozen"
