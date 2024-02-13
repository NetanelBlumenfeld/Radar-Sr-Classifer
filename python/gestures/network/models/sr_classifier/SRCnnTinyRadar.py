import torch
from gestures.network.models.basic_model import BasicModel

EPSILON = 1e-8


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
        _, _, _, channels, H, W = high_res_imgs.size()
        high_res_imgs = high_res_imgs.reshape(-1, channels, H, W)
        batch = batch.permute(1, 0, 2, 3, 4, 5).to(device)
        true_label = true_label.permute(1, 0).to(device)

        return batch, [high_res_imgs, true_label]

    def forward(self, inputs):

        # Assuming inputs is of shape [sequence_length, batch_size, sensors, channels, H, W]
        sequence_length, batch_size, sensors, channels, H, W = inputs.size()

        # Reshape inputs to combine sequence_length and batch_size for batch processing
        inputs = inputs.reshape(-1, channels, H, W)

        # Apply super resolution in a batched manner
        inputs = self.drln(inputs)

        # Reshape rec_imgs back to separate sequence_length and batch_size, adjusting for scale_factor
        rec_imgs = inputs.reshape(
            sequence_length,
            batch_size,
            sensors,
            channels,
            H * self.scale_factor,
            W * self.scale_factor,
        )

        real_part = rec_imgs[:, :, :, 0, :, :]
        imaginary_part = rec_imgs[:, :, :, 1, :, :]

        del rec_imgs

        # Perform FFT on the complex tensor along the spatial dimension (assuming last dimension here)
        # Note: Adjust the dim parameter based on how your data is structured and where FFT should be applied
        doppler_maps = torch.abs(
            torch.fft.fftshift(
                torch.fft.fft(torch.complex(real_part, imaginary_part), dim=-2), dim=-2
            )
        )

        # # Normalize the Doppler maps
        # # Compute the min and max values for normalization across the spatial dimensions
        # min_vals, _ = torch.min(doppler_maps, dim=-1, keepdim=True)
        # min_vals, _ = torch.min(min_vals, dim=-2, keepdim=True)
        # max_vals, _ = torch.max(doppler_maps, dim=-1, keepdim=True)
        # max_vals, _ = torch.max(max_vals, dim=-2, keepdim=True)

        # # Apply normalization using broadcasting
        # EPSILON = 1e-8  # Small constant to avoid division by zero
        # normalized_doppler_maps = (doppler_maps - min_vals) / (
        #     max_vals - min_vals + EPSILON
        # )

        # Apply classifier on Doppler maps (assuming classifier can handle the batched format)
        y_labels_pred = self.classifier(doppler_maps)

        return inputs, y_labels_pred  # Adjust based on actual processing steps
