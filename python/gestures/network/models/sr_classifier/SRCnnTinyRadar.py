import torch
from gestures.network.models.basic_model import BasicModel
from gestures.network.models.super_resolution.down_sample_net import DownsampleConv2d
from gestures.utils_processing_data import (
    DopplerMapBatch,
    NormalizeBatch,
    RealToComplexBatch,
)

EPSILON = 1e-8


class CombinedSRDrlnClassifier(BasicModel):
    def __init__(
        self,
        sr: BasicModel,
        classifier: BasicModel,
        scale_factor: int = 4,
    ):
        model_name = f"sr_{sr.model_name}_classifier_{classifier.model_name}"
        super(CombinedSRDrlnClassifier, self).__init__(model_name)
        self.drln = sr
        self.classifier = classifier
        self.scale_factor = scale_factor
        self.data_processor = torch.nn.Sequential(
            RealToComplexBatch(), NormalizeBatch(), DopplerMapBatch()
        ).to(torch.device("cpu"))
        self.down_sample_net = DownsampleConv2d()

    @staticmethod
    def reshape_to_model_output(batch, labels, device):
        high_res_imgs, true_label = labels
        high_res_imgs = high_res_imgs.permute(1, 0, 2, 3, 4, 5).to(device)
        sequence_length, batch_size, sensors, channels, H, W = high_res_imgs.size()
        new_batch = sequence_length * batch_size * sensors
        high_res_imgs = high_res_imgs.reshape(new_batch, channels, H, W)
        batch = batch.permute(1, 0, 2, 3, 4, 5).to(device)
        true_label = true_label.permute(1, 0).to(device)

        return batch, [high_res_imgs, true_label]

    def forward(self, inputs):

        # Assuming inputs is of shape [sequence_length, batch_size, sensors, channels, H, W]
        sequence_length, batch_size, sensors, channels, H, W = inputs.size()
        new_batch = sequence_length * batch_size * sensors
        inputs = inputs.reshape(new_batch, channels, H, W)

        # Apply super resolution in a batched manner
        inputs = self.drln(inputs)
        inputs = self.down_sample_net(inputs)

        # Reshape rec_imgs back to separate sequence_length and batch_size, adjusting for scale_factor
        rec_imgs = inputs.reshape(
            sequence_length,
            batch_size,
            sensors,
            channels,
            32,
            492,
        )

        doppler_maps = self.data_processor(rec_imgs)

        y_labels_pred = self.classifier(doppler_maps)

        return inputs, y_labels_pred  # Adjust based on actual processing steps
