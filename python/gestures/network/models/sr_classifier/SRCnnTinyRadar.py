import torch
from gestures.network.models.basic_model import BasicModel
from numpy import rec
from torch.fft import fft, fftshift


class CombinedSRCNNClassifier(BasicModel):
    def __init__(
        self,
        srcnn: BasicModel,
        classifier: BasicModel,
        scale_factor: int = 1,
    ):
        model_name = f"sr_{srcnn.model_name}_classifier_{classifier.model_name}"
        super(CombinedSRCNNClassifier, self).__init__(model_name)
        self.srcnn = srcnn
        self.classifier = classifier
        self.scale_factor = scale_factor

    # @staticmethod
    # def reshape_to_model_output(batch, labels, device):
    #     high_res_imgs, true_label = labels
    #     high_res_imgs = high_res_imgs.permute(1, 0, 2, 3, 4).to(device)
    #     batch = batch.permute(1, 0, 2, 3, 4).to(device)
    #     true_label = true_label.permute(1, 0).to(device)

    #     return batch, [high_res_imgs, true_label]

    def forward(self, inputs):
        sequence_length, batch_size, channels, H, W = inputs.size()

        # Process each sequence element with self.srcnn
        processed_sequence = []
        for i in range(sequence_length):
            # Extract the sequence element and add a channel dimension
            x = inputs[i].reshape(batch_size * channels, 1, H, W)

            # Apply srcnn
            rec_img = self.srcnn(x)

            # Remove the channel dimension and add it to the processed list
            processed_sequence.append(
                rec_img.reshape(
                    batch_size, channels, H * self.scale_factor, W * self.scale_factor
                )
            )

        # Recombine the sequence
        rec_img = torch.stack(processed_sequence, dim=0)

        # Apply the classifier
        y_labels_pred = self.classifier(rec_img)
        return rec_img, y_labels_pred


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

    @staticmethod
    def reshape_to_model_output(batch, labels, device):
        high_res_imgs, true_label = labels
        high_res_imgs = high_res_imgs.permute(1, 0, 2, 3, 4, 5).to(device)
        batch = batch.permute(1, 0, 2, 3, 4, 5).to(device)
        true_label = true_label.permute(1, 0).to(device)

        return batch, [high_res_imgs, true_label]

    def forward(self, inputs):
        sequence_length, batch_size, sensors, channels, H, W = inputs.size()

        # Process each sequence element with self.drln
        doppler_maps, rec_imgs = [], []
        for i in range(sequence_length):
            # Extract the sequence element and add a channel dimension
            x = inputs[i].reshape(batch_size * sensors, channels, H, W)

            # Apply super resolution
            rec_img = self.drln(x)
            rec_img = rec_img.reshape(
                batch_size,
                sensors,
                channels,
                H * self.scale_factor,
                W * self.scale_factor,
            )
            rec_imgs.append(rec_img)
            # convert to doppler map
            doppler_map = rec_img[:, :, 0] + 1j * rec_img[:, :, 1]
            doppler_map = torch.abs(fftshift(fft(doppler_map, dim=2), dim=2))

            # Remove the channel dimension and add it to the processed list
            doppler_maps.append(doppler_map)

        # Recombine the sequence
        doppler_maps = torch.stack(doppler_maps, dim=0)
        rec_imgs = torch.stack(rec_imgs, dim=0)

        # Apply the classifier
        y_labels_pred = self.classifier(doppler_maps)
        return rec_imgs, y_labels_pred
